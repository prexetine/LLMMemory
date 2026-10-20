"""本地服务初始化/健康检查/启动/停止；不启动用户提取。"""
import argparse
import json
import secrets
import subprocess
from pathlib import Path
from agent_retrieval import LocalEncoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['init','start','health','stop'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--python')
    parser.add_argument('--model')
    args=parser.parse_args()
    path=Path(args.config).resolve()
    if args.action=='init':
        if path.exists(): raise ValueError('配置已存在，禁止覆盖服务令牌')
        if not args.python or not args.model: raise ValueError('init需要--python和--model')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(python=str(Path(args.python).resolve(strict=True)),model_path=str(Path(args.model).resolve(strict=True)),
                                        port=8766,token=secrets.token_urlsafe(32),batch_size=4,chunk_tokens=512,threads=4),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('本地配置已生成（令牌不显示）'); return
    cfg=json.loads(path.read_text(encoding='utf-8'))
    local=dict(port=cfg['port'],service_config=str(path),timeout_seconds=10)
    if args.action=='start':
        log=path.parent/'bge-service.log'
        with log.open('a',encoding='utf-8') as output:
            process=subprocess.Popen([cfg['python'],'-X','utf8',str(Path(__file__).with_name('bge_service.py')),'--config',str(path)],
                                     stdout=output,stderr=output,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(json.dumps(dict(pid=process.pid,log=str(log),status='starting_check_health_before_use'),ensure_ascii=False))
    else:
        client=LocalEncoder(local)
        print(json.dumps(client.request('/shutdown',{}) if args.action=='stop' else client.info,ensure_ascii=False))


if __name__=='__main__': main()
