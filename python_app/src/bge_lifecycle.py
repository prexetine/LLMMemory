"""批次级本地编码服务租约；Windows内核锁随进程退出自动释放。"""
from __future__ import annotations

import json
import logging
import msvcrt
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from agent_retrieval import LocalEncoder, RetrievalError


def lock_file(path: Path):
    handle = path.open('a+', encoding='utf-8')
    handle.seek(0, 2)
    if not handle.tell():
        handle.write('0')
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise
    return handle


@contextmanager
def guard(path: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        try:
            handle = lock_file(path)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError('等待BGE生命周期锁超时')
            time.sleep(0.2)
    try:
        yield
    finally:
        handle.close()


def active_leases(folder: Path) -> int:
    active = 0
    for path in folder.glob('*.lease'):
        try:
            handle = lock_file(path)
        except OSError:
            active += 1  # 无法证明空闲时，保守地不停止服务。
        else:
            handle.close()
            path.unlink()
    return active


def probe(local: dict):
    try:
        return LocalEncoder(local)
    except RetrievalError:
        return None


def launch(config: dict, path: Path):
    with (path.parent / 'bge-service.log').open('a', encoding='utf-8') as output:
        return subprocess.Popen(
            [config['python'], '-X', 'utf8', str(Path(__file__).with_name('bge_service.py')), '--config', str(path)],
            stdout=output, stderr=output, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def check_service(client, cfg: dict) -> None:
    info = client.info
    if (info.get('status') != 'ready' or
            str(Path(info.get('model_path', '')).resolve()).casefold() != str(Path(cfg['model_path']).resolve()).casefold() or
            info.get('chunk_tokens') != cfg.get('chunk_tokens', 512)):
        raise RuntimeError('BGE服务模型配置不匹配或为旧版服务，请先手动停止旧服务后重试')


@contextmanager
def managed_encoder(retrieval: dict):
    """覆盖整个批次及导出；最后一个租约退出后停止服务。"""
    path = Path(retrieval['service_config']).resolve(strict=True)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg['port'] != retrieval.get('port', 8766):
        raise ValueError('retrieval.port与服务配置不一致')
    timeout = retrieval.get('startup_timeout_seconds', 180)
    if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
        raise ValueError('startup_timeout_seconds须为0至3600秒之间的正数')
    local = dict(retrieval, timeout_seconds=2)
    folder = path.parent / (path.stem + '-leases')
    folder.mkdir(exist_ok=True)
    gate = folder / 'lifecycle.lock'
    lease = None
    process = None
    client = None
    try:
        with guard(gate, timeout + 30):
            active_leases(folder)
            lease = lock_file(folder / (uuid.uuid4().hex + '.lease'))
            client = probe(local)
            if client is None:
                logging.info('正在启动并加载本地BGE模型，最长等待%s秒；日志=%s', timeout, path.parent / 'bge-service.log')
                process = launch(cfg, path)
                deadline = time.monotonic() + timeout
                while client is None:
                    if process.poll() is not None:
                        raise RuntimeError(f'BGE启动进程退出，日志：{path.parent / "bge-service.log"}')
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'BGE模型加载超时，日志：{path.parent / "bge-service.log"}')
                    time.sleep(0.5)
                    client = probe(local)
            check_service(client, cfg)
            logging.info('本地BGE就绪 dimension=%s；批次结束后自动释放服务', client.info['dimension'])
        yield client.info
    finally:
        if lease is not None:
            with guard(gate, timeout + 30):
                lease.close()
                if active_leases(folder):
                    logging.info('其他批次仍在使用BGE，本批次仅释放租约')
                else:
                    try:
                        current = probe(local)
                        if current is not None:
                            check_service(current, cfg)
                            current.request('/shutdown', {})
                            deadline = time.monotonic() + 15
                            while probe(local) is not None:
                                if time.monotonic() >= deadline:
                                    raise RuntimeError('BGE停止超时，请检查服务日志')
                                time.sleep(0.2)
                            logging.info('最后一个批次结束，本地BGE服务已停止')
                        elif process is not None and process.poll() is None:
                            # Windows虚拟环境启动器还有子进程，仅清理本次创建的树。
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                           capture_output=True, timeout=15, check=True)
                    except Exception:
                        logging.exception('BGE退出清理失败；请使用bge_control.py stop检查并释放服务')
