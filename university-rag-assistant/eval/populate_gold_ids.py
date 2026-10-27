import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; IP=ROOT/"data/index/chunks.json"; QP=ROOT/"data/eval/questions.jsonl"
def main():
    if not IP.exists(): raise SystemExit("Build index first: python -m src.build_index")
    chunks=json.loads(IP.read_text(encoding="utf-8")); cases=[json.loads(x) for x in QP.read_text(encoding="utf-8").splitlines() if x.strip()]
    for c in cases:
        matches=[x for x in chunks if x["source_doc"]==c["source_doc"] and c["source_passage"].lower() in x["text"].lower()]
        if not matches: matches=[x for x in chunks if x["source_doc"]==c["source_doc"]]
        c["relevant_chunk_ids"]=[matches[0]["chunk_id"]] if matches else []
    QP.write_text("\n".join(json.dumps(c,ensure_ascii=False) for c in cases)+"\n",encoding="utf-8")
    print("Wrote provisional gold IDs; manually review before reporting results.")
if __name__=="__main__":main()
