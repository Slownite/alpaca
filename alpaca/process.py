"""PID tracking and process control (ps/stop) for running servers."""
from __future__ import annotations
from alpaca.path import path_helper
import psutil
import json
import pathlib

type Meta = dict[str, str | int | bool]

def terminate_running_process(pid: int, timeout: float = 10.0)->None:
    try:
        p =  psutil.Process(pid)
        p.terminate()
        p.wait(timeout)
    except psutil.TimeoutExpired:
        p.kill()
    
    
def cleanup(p: pathlib.Path)->None:
    p.unlink(missing_ok=True)
    
def record(meta: Meta) -> pathlib.Path:
    name = meta["name"] if meta.get("name") else meta["model"]
    meta["name"] = name
    path = path_helper("run/")
    meta_json_file = path / f"{name}.json"
    with open(meta_json_file, "w") as f:
        json.dump(meta, f)
    return path

def list_running()->list[Meta]:
    path = path_helper("run/")
    result = []
    for p in path.glob("*.json"):
        with open(p, "r") as f:
            file: Meta = json.load(f)
            pid = file["pid"]
            if not psutil.pid_exists(pid):
                cleanup(p)
            else:
               result.append(file)
    return result

def find(name: str)->Meta | None:
    metas = list_running()
    for m in metas:
        if m.get("name") == name:
            return m
    return None

def stop(name: str, timeout: float = 10.0) -> bool:
    meta = find(name)
    if meta is None:
        return False
    pid = meta["pid"]
    if psutil.pid_exists(pid):
        terminate_running_process(pid)
        path = path_helper("run/")
        cleanup(path / f"{name}.json")
        return True
    return False
        
        

