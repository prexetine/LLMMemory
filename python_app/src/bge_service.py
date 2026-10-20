"""仅回环访问的单进程BGE-M3稠密编码服务；不读取用户数据库。"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def memory_usage():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    psapi=ctypes.WinDLL('psapi'); fn=psapi.GetProcessMemoryInfo
    fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not fn(wintypes.HANDLE(-1),ctypes.byref(counters),counters.cb): return {}
    return dict(working_set_bytes=counters.WorkingSetSize,peak_working_set_bytes=counters.PeakWorkingSetSize)


class Encoder:
    def __init__(self, model_path: str, batch_size: int = 4, chunk_tokens: int = 512, threads: int = 4):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(threads)
        started = time.perf_counter()
        path = Path(model_path).resolve(strict=True)
        fingerprint = hashlib.sha256()
        for name in ('config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json'):
            with (path / name).open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    fingerprint.update(block)
        self.signature = 'bge-dense-cls-chunk-v1:' + fingerprint.hexdigest() + ':' + str(chunk_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval()
        self.batch_size, self.chunk_tokens = batch_size, chunk_tokens
        self.dimension = self.model.config.hidden_size
        self.load_seconds = time.perf_counter() - started
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def encode(self, texts: list[str]) -> list[list[float]]:
        torch = self.torch
        chunks=[]
        for owner,text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            chunks.extend((owner,ids[i:i+self.chunk_tokens]) for i in range(0,len(ids),self.chunk_tokens))
            if not ids: chunks.append((owner,[]))
        sums=torch.zeros((len(texts),self.dimension))
        # 跨文本按长度分桶组批，减少padding；长文本所有片段均参与聚合。
        chunks.sort(key=lambda item:len(item[1]))
        for start in range(0,len(chunks),self.batch_size):
            group=chunks[start:start+self.batch_size]
            items=[self.tokenizer.prepare_for_model(c,add_special_tokens=True,return_attention_mask=True) for _,c in group]
            batch=self.tokenizer.pad(items,padding=True,return_tensors='pt')
            with torch.inference_mode():
                encoded=self.model(**batch).last_hidden_state[:,0]
                vectors=torch.nn.functional.normalize(encoded,p=2,dim=1)
            for (owner,ids),vector in zip(group,vectors): sums[owner]+=vector*max(1,len(ids))
        return torch.nn.functional.normalize(sums,p=2,dim=1).tolist()


def serve(config: dict) -> None:
    encoder = Encoder(config['model_path'], config.get('batch_size', 4), config.get('chunk_tokens', 512), config.get('threads', 4))
    token = config['token']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logging.info(fmt, *args)

        def send(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {'error': 'not found'})
            self.send(200, dict(status='ready', model_signature=encoder.signature, dimension=encoder.dimension,
                                model_path=str(Path(config['model_path']).resolve()), chunk_tokens=config.get('chunk_tokens', 512),
                                load_seconds=encoder.load_seconds, parameters=encoder.parameters, device='cpu', pid=os.getpid(),**memory_usage()))

        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer ' + token:
                return self.send(403, {'error': 'unauthorized'})
            if self.path == '/shutdown':
                self.send(200, {'status': 'stopping'})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != '/encode':
                return self.send(404, {'error': 'not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1048576:
                    raise ValueError('request size')
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                if set(data) != {'texts'} or not isinstance(data['texts'], list) or not 1 <= len(data['texts']) <= 64:
                    raise ValueError('texts must contain 1..64 strings')
                if any(not isinstance(t, str) or not t.strip() or len(t) > 100000 for t in data['texts']):
                    raise ValueError('invalid text')
                start = time.perf_counter()
                vectors = encoder.encode(data['texts'])
                elapsed = time.perf_counter() - start
                logging.info('编码完成 texts=%s elapsed=%.3fs', len(vectors), elapsed)
                self.send(200, dict(vectors=vectors, model_signature=encoder.signature, dimension=encoder.dimension, elapsed_seconds=elapsed))
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, {'error': str(exc)})
            except Exception as exc:
                logging.exception('编码失败')
                self.send(500, {'error': type(exc).__name__})
    server = HTTPServer(('127.0.0.1', config.get('port', 8766)), Handler)
    logging.info('BGE ready port=%s load=%.2fs', server.server_port, encoder.load_seconds)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Windows文件锁在加载模型前取得；重复启动不再多加载一份权重。
    import msvcrt
    with Path(args.config).with_suffix('.lock').open('a+',encoding='utf-8') as lock:
        lock.seek(0,2)
        if lock.tell()==0:
            lock.write('0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            serve(json.loads(Path(args.config).read_text(encoding='utf-8')))
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
