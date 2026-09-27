import React, { useEffect, useState } from "react";
type Summary = { data_type: string; validation: Record<string, number>; test_counts: Record<string, number> };
const api = (path: string) => fetch(`http://localhost:8000${path}`).then(r => r.json());
export default function Dashboard() {
 const [data, setData] = useState<Summary | null>(null); useEffect(() => { api("/api/dataset/stats").then(setData); }, []);
 if (!data) return <main className="p-8 text-slate-300 bg-slate-950">Loading pipeline data…</main>;
 const demo=data.data_type === "synthetic";
 return <main className="min-h-screen p-8 bg-slate-950 text-white"><header><p className="text-cyan-400 font-semibold">ENTITY RESOLUTION / DASHBOARD</p>{demo && <aside className="mt-4 rounded border border-amber-500 p-4 text-amber-200"><b>DEMO MODE</b><br/>Real challenge dataset not detected. Running with synthetic development data. Add TSV files to dataset/train/ and dataset/test/.</aside>}</header><section className="grid grid-cols-2 gap-4 mt-8">{Object.entries({...data.test_counts,...data.validation}).map(([k,v])=><article className="rounded bg-slate-900 p-5" key={k}><small>{k.replaceAll("_"," ")}</small><strong className="block text-2xl mt-1">{typeof v === "number" ? v.toFixed(3) : v}</strong></article>)}</section><nav className="mt-8 text-cyan-300">/dashboard · /dataset · /blocking · /matching · /evaluation · /results</nav></main>;
}
