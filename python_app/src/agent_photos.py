"""受限图片来源、固定视觉证据快照与照片Session；不把图像当作用户原话。"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

from agent_data import digest, exact_keys, load_user_roster

PHOTO_POLICY = "photo-observation-v1"


def options(config: dict) -> dict | None:
    value = config.get("photos")
    if value is None:
        return None
    exact_keys(value, {"enabled", "allowed_hosts", "max_bytes", "max_pixels", "max_side", "input_token_reserve"})
    if type(value["enabled"]) is not bool:
        raise ValueError("photos.enabled必须为布尔值")
    if not value["enabled"]:
        return None
    # 官方能力表：https://docs.volcengine.com/docs/ark/model-list?lang=zh（2026-10-08核对）
    # 执行版本由入口统一验证；图片组件只验证模型和图片安全能力。
    if config.get("provider") != "Doubao" or config.get("model_alias") not in ("Doubao-Seed-2.0-lite", "Doubao-Seed-2.0-pro", "Doubao-Seed-2.1-lite"):
        raise ValueError("图片链路仅允许豆包Seed2.0 Lite/Pro、2.1 Lite")
    if not isinstance(value["allowed_hosts"], list) or not value["allowed_hosts"] or any(h != "hipanda-cdn-ws.toycloud.com" for h in value["allowed_hosts"]):
        raise ValueError("图片下载域名必须为已核对的日志CDN")
    for field in ("max_bytes", "max_pixels", "max_side", "input_token_reserve"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError("图片容量参数必须为正整数")
    if value["max_bytes"] > 20000000 or value["max_pixels"] > 40000000 or value["max_side"] > 1280 or value["input_token_reserve"] < 4096:
        raise ValueError("图片配置超过安全边界或视觉token预留不足")
    return value


def safe_url(url: str, settings: dict) -> str:
    p = urllib.parse.urlsplit(url)
    if (p.scheme not in ("https", "http") or p.hostname not in settings["allowed_hosts"]
            or p.username or p.password or p.port not in (None, 443) or p.fragment
            or any(ord(c) < 32 for c in url)):
        raise ValueError("图片地址不在允许范围")
    # 已确认CDN的历史HTTP地址只升级HTTPS，不回退明文协议。
    return urllib.parse.urlunsplit(("https", p.netloc, p.path, p.query, ""))


def load_photo_sessions(root: Path, device: str, config: dict) -> tuple[list[dict], dict]:
    settings = options(config)
    if settings is None:
        return [], {}
    if device not in {r["device_id"] for r in load_user_roster(root)}:
        raise ValueError("照片设备不在当前名单")
    result, excluded, seen, urls = [], {}, {}, set()
    with gzip.open(root / "candidate_data" / device / "photo.jsonl.gz", "rt", encoding="utf-8") as stream:
        for item in map(json.loads, stream):
            if item["disposition"] != "eligible":
                reason = item["disposition"]
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            data = item["data"]
            if data.get("device_id") != device or str(data.get("del_flag")) not in ("0", "0.0") or not data.get("picture_id"):
                raise ValueError("照片归属、删除标记或主键无效")
            url = safe_url(data["picture_url"], settings)
            eid = f"{device}:photo:{data['picture_id']}"
            fp = digest(data)
            if eid in seen:
                if seen[eid] != fp:
                    raise ValueError("同照片ID内容冲突")
                excluded["duplicate_picture_id"] = excluded.get("duplicate_picture_id", 0) + 1
                continue
            seen[eid] = fp
            if url in urls:
                excluded["duplicate_picture_url"] = excluded.get("duplicate_picture_url", 0) + 1
                continue
            urls.add(url)
            marker = f"[图片证据 {eid}；不是用户原话，事实须核对附带画面]"
            turn = dict(evidence_id=eid, source_kind="photo", photo_policy=PHOTO_POLICY,
                        user_text=marker, assistant_text="", occurred_at=item["occurred_at"],
                        source_file=item["source_file"], sheet=item["sheet"], source_row=item["row"])
            result.append(dict(device_id=device, session_id="photo-" + digest(eid)[:24],
                start=item["occurred_at"], end=item["occurred_at"], turns=[turn],
                input_hash=digest([PHOTO_POLICY, item]), source_kind="photo", image_url=url))
    result.sort(key=lambda s: (s["start"], s["session_id"]))
    return result, dict(photo_sessions=len(result), photo_excluded=excluded, photo_policy=PHOTO_POLICY)


def normalize_image(raw: bytes, settings: dict) -> bytes:
    from PIL import Image, ImageOps
    if len(raw) > settings["max_bytes"]:
        raise ValueError("图片字节超限")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in ("JPEG", "PNG", "WEBP") or image.width * image.height > settings["max_pixels"]:
            raise ValueError("图片格式或像素超限")
        image.load()
        frame = ImageOps.exif_transpose(image).convert("RGB")
        frame.thumbnail((settings["max_side"], settings["max_side"]))
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


def bind_image(session: dict, out: Path, config: dict, client: object) -> None:
    """一次获取并冻结去EXIF快照，续跑只使用同一哈希快照；不将Base64写进证据或日志。"""
    settings = options(config)
    if settings is None:
        raise ValueError("未启用图片处理")
    cache = out / "photo_assets"
    cache.mkdir(exist_ok=True)
    key = digest([session["input_hash"], PHOTO_POLICY, settings])
    jpg, metadata = cache / (key + ".jpg"), cache / (key + ".json")
    old_root = config.get("photo_cache_source")
    if not metadata.exists() and old_root:
        prior = Path(old_root)
        if not prior.is_absolute():
            prior = Path(__file__).resolve().parents[2] / prior
        prior = prior / session["device_id"] / "photo_assets"
        old_meta, old_jpg = prior / metadata.name, prior / jpg.name
        if old_meta.exists():
            m = json.loads(old_meta.read_text(encoding="utf-8"))
            b = old_jpg.read_bytes()
            if m.get("input_hash") != session["input_hash"] or m.get("image_sha256") != hashlib.sha256(b).hexdigest():
                raise ValueError("旧图片快照校验失败，禁止静默替换图片")
            jpg.write_bytes(b)
            metadata.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8", newline="\n")
    if metadata.exists():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        raw = jpg.read_bytes()
        if meta["image_sha256"] != hashlib.sha256(raw).hexdigest() or meta["input_hash"] != session["input_hash"]:
            raise ValueError("图片证据缓存损坏，禁止静默换图")
    else:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = urllib.request.build_opener(NoRedirect())
        request = urllib.request.Request(safe_url(session["image_url"], settings), headers={"User-Agent": "ToyMemory-photo/1.0"})
        try:
            with opener.open(request, timeout=config["timeout_seconds"]) as response:
                original = response.read(settings["max_bytes"] + 1)
        except Exception as exc:
            raise RuntimeError("图片获取失败：" + type(exc).__name__) from None
        raw = normalize_image(original, settings)
        meta = dict(image_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=hashlib.sha256(original).hexdigest(), input_hash=session["input_hash"])
        jpg.write_bytes(raw)
        metadata.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    turn = session["turns"][0]
    turn["image_sha256"] = meta["image_sha256"]
    turn["image_asset"] = "photo_assets/" + jpg.name
    client.image_assets = {turn["evidence_id"]: dict(sha256=meta["image_sha256"], data_url="data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"))}


def image_content(client: object, data: object) -> list[dict]:
    """只为明确带图片来源及哈希的证据包附图；图片缺失时不能退化为纯文本审核。"""
    found = {}
    def walk(value):
        if isinstance(value, dict):
            if value.get("source_kind") == "photo" and "evidence_id" in value:
                eid = value["evidence_id"]
                asset = getattr(client, "image_assets", {}).get(eid)
                if not asset or asset["sha256"] != value.get("image_sha256"):
                    raise ValueError("视觉证据未绑定或哈希不一致")
                found[eid] = asset
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(data)
    if len(found) > 1:
        raise ValueError("当前图片链路每次仅允许一张证据图")
    result = []
    for eid, asset in found.items():
        result.extend([{"type": "text", "text": "图片证据ID=" + eid},
                       {"type": "image_url", "image_url": {"url": asset["data_url"], "detail": "high"}}])
    return result
