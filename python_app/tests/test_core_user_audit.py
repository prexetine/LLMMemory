"""阶段一筛选的边界、流式读取和排序测试。"""
import json
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core_user_audit import StreamBook, classify, devices_for, parse_time, rank_stats, scan
from core_user_report import make_report

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config/core_user_audit.json').read_text(encoding='utf-8'))
D = '12354838B667A738'


class CoreAuditTests(unittest.TestCase):
    def test_end_to_end_counts_provenance_and_no_overwrite(self) -> None:
        ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        d2 = '12354838B667A739'
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            inputs = base / 'input'
            out = base / 'output'
            inputs.mkdir()
            out.mkdir()
            dialogue = [dict(self.dialogue(), dialog_id='1'), dict(self.dialogue(), dialog_id='1'),
                        dict(self.dialogue(), dialog_id='2'), dict(self.dialogue(device_id=d2), dialog_id='3')]
            sources = {
                '智能体对话': dialogue,
                '拍照': [dict(device_id=D, user_id='P1', picture_id='1', create_time='46242.5', del_flag='0', picture_url='https://example.com/p.png')],
                '通话': [dict(video_id='1', caller_type='device', caller_id=D, callee_type='device', callee_id=d2, started_at='46242.5', connected_at='46242.50001', ended_at='46242.50002', state='ENDED')],
                '定位': [dict(id='1', device_id=D, timestamp='1786000000000', lat='22', lon='114', accuracy='5')],
            }
            for label, records in sources.items():
                headers = list(records[0])
                rows = [headers] + [[r.get(h) for h in headers] for r in records]
                xml_rows = []
                for number, values in enumerate(rows, 1):
                    cells = ''.join(f'<c r="{chr(65+i)}{number}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>' for i,v in enumerate(values) if v is not None)
                    xml_rows.append(f'<row r="{number}">{cells}</row>')
                with zipfile.ZipFile(inputs / ('J20' + label + '.xlsx'), 'w') as z:
                    z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="data" r:id="r1"/></sheets></workbook>')
                    z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
                    z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
            config = dict(CONFIG, candidate_count=2, core_count=1)
            scan(inputs, out, config)
            make_report(out)
            summary = json.loads((out / 'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['audit']['dialogue']['dispositions']['duplicate'], 1)
            self.assertEqual(summary['audit']['dialogue']['dispositions']['eligible'], 3)
            self.assertTrue(json.loads((out / 'validation.json').read_text(encoding='utf-8'))['passed'])
            with self.assertRaises(FileExistsError):
                scan(inputs, out, config)

    def test_excel_and_epoch(self) -> None:
        dt, kind = parse_time('46242.5')
        self.assertEqual(dt.hour, 12)
        self.assertEqual(kind, 'excel_serial_assumed_local')
        back, _ = parse_time(str(dt.timestamp() * 1000))
        self.assertEqual(dt, back)
        self.assertEqual(parse_time('bad'), (None, 'invalid'))

    def dialogue(self, **overrides) -> dict:
        row = dict(device_id=D, create_time='46242.5', status='0', del_flag='0', text='不是')
        row.update(overrides)
        return row

    def test_dialogue_roles_and_short_correction(self) -> None:
        self.assertEqual(classify('dialogue', self.dialogue(), CONFIG)[0], 'eligible')
        self.assertEqual(classify('dialogue', self.dialogue(text=None, content='你好'), CONFIG)[0], 'no_user_text')
        self.assertEqual(classify('dialogue', self.dialogue(text=None, audio='x'), CONFIG)[0], 'audio_only')
        self.assertEqual(classify('dialogue', self.dialogue(del_flag='1'), CONFIG)[0], 'deleted_or_unknown_delete')
        self.assertEqual(classify('dialogue', self.dialogue(device_id=None), CONFIG)[0], 'missing_device')

    def test_test_phrase_does_not_remove_event(self) -> None:
        status, flags, _ = classify('dialogue', self.dialogue(text='测试测试'), CONFIG)
        self.assertEqual(status, 'eligible')
        self.assertIn('test_phrase', flags)

    def test_photo_asset(self) -> None:
        row = dict(device_id=D, create_time='46242.5', del_flag='0', picture_url='https://example.com/album_enter_123.png')
        self.assertEqual(classify('photo', row, CONFIG)[0], 'system_asset_suspect')

    def test_call_connected_not_completed_label(self) -> None:
        row = dict(caller_type='app', caller_id='P1', callee_type='device', callee_id=D,
                   started_at='46242.5', state='ENDED', end_reason='completed', duration='0')
        self.assertEqual(devices_for('call', row), [D])
        self.assertEqual(classify('call', row, CONFIG)[0], 'not_connected')
        row.update(connected_at='46242.50001', ended_at='46242.50002')
        self.assertEqual(classify('call', row, CONFIG)[0], 'eligible')

    def test_location_not_ranking(self) -> None:
        stats = {d: {'counts': Counter(dialogue_eligible=n, location_raw=loc), 'days': {'2026-08-01'}}
                 for d, n, loc in [('A', 1, 999999), ('B', 2, 0)]}
        self.assertEqual(rank_stats(stats)[0]['device_id'], 'B')

    def test_stream_sparse_cells_and_second_sheet(self) -> None:
        ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            with zipfile.ZipFile(p / 'a.xlsx', 'w') as z:
                z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}"><si><t>device_id</t></si><si><t>text</t></si></sst>')
                z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="one" r:id="r1"/><sheet name="two" r:id="r2"/></sheets></workbook>')
                z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/><Relationship Id="r2" Target="worksheets/sheet2.xml"/></Relationships>')
                for i in (1, 2):
                    z.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="B2" t="inlineStr"><is><t>不是</t></is></c></row></sheetData></worksheet>')
            book = StreamBook(p / 'a.xlsx', p / 'cache.sqlite3')
            rows = list(book.rows())
            book.close()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][2], {'device_id': None, 'text': '不是'})
            self.assertEqual(rows[1][0], 'two')


if __name__ == '__main__':
    unittest.main()
