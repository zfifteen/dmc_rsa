import React, { useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Copy, Hammer, Play, Repeat, Rocket, RotateCcw } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// ============================
// Math & Sampling Utilities
// ============================

const PHI = (1 + Math.sqrt(5)) / 2; // golden ratio

// Deterministic xorshift32 PRNG for MC baseline & CP shifts
function createXorShift32(seed: number) {
  let s = seed >>> 0 || 0x9e3779b9;
  function nextU32() {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17; s >>>= 0;
    s ^= s << 5;  s >>>= 0;
    return s >>> 0;
  }
  return {
    next: () => nextU32() / 4294967296,
    nextInt: (maxExclusive: number) => (nextU32() % maxExclusive) >>> 0,
    seed: (v: number) => { s = v >>> 0; }
  };
}

// Van der Corput base-b radical inverse
function vdc(n: number, base: number) {
  let v = 0;
  let denom = 1;
  while (n > 0) {
    denom *= base;
    v += (n % base) / denom;
    n = Math.floor(n / base);
  }
  return v;
}

// 2D Halton sequence (bases 2,3). We start at index+1 to avoid 0.
function halton2(i: number) {
  const n = i + 1;
  return [vdc(n, 2), vdc(n, 3)];
}

// BigInt integer sqrt via Newton's method
function isqrt(n: bigint): bigint {
  if (n < 0n) throw new Error("isqrt of negative");
  if (n < 2n) return n;
  // Initial guess: 1 << (bitlen(n)/2)
  let x0 = 1n << (BigInt((n.toString(2).length + 1) >> 1));
  let x1 = (x0 + n / x0) >> 1n;
  while (x1 < x0) {
    x0 = x1;
    x1 = (x0 + n / x0) >> 1n;
  }
  return x0;
}

// Constrain candidate to [2, sqrtN]
function clampCandidate(c: bigint, sqrtN: bigint): bigint {
  const lo = 2n;
  const hi = sqrtN < lo ? lo : sqrtN;
  if (c < lo) return lo;
  if (c > hi) return hi;
  return c;
}

// Map u∈[0,1) to candidate in [2, sqrtN] linearly
function linearMapToCandidate(u: number, sqrtN: bigint): bigint {
  const span = Number(sqrtN - 2n);
  const t = Math.min(0.999999999999, Math.max(0, u));
  // For very large sqrtN this cast may lose precision; this demo targets moderate N.
  const off = Math.floor(t * Math.max(0, span));
  return 2n + BigInt(off);
}

// φ-biased map concentrating mass near the upper end (≈ sqrtN)
// Uses exponent α (default φ). u' = u^(1/α) skews samples toward 1.
function phiBiasedToCandidate(u: number, sqrtN: bigint, alpha = PHI): bigint {
  const t = Math.pow(Math.min(0.999999999999, Math.max(0, u)), 1 / Math.max(1e-6, alpha));
  return linearMapToCandidate(t, sqrtN);
}

// Simple histogram between [2, sqrtN] using fixed bin count
function buildHistogram(values: bigint[], sqrtN: bigint, bins = 40) {
  if (values.length === 0) return [] as { bin: string; count: number }[];
  const lo = 2n;
  const hi = sqrtN;
  const width = Number(hi - lo);
  const binWidth = Math.max(1, Math.floor(width / bins));
  const counts = new Array(Math.min(bins, Math.max(1, Math.floor(width / binWidth)))).fill(0);
  for (const v of values) {
    const idx = Math.min(counts.length - 1, Math.floor(Number(v - lo) / binWidth));
    counts[idx]++;
  }
  return counts.map((c, i) => ({ bin: `${Number(lo) + i * binWidth}`, count: c }));
}

// Metrics type
type TrialMetrics = {
  mode: string;
  samples: number;
  timeMs: number;
  cps: number; // candidates per second
  unique: number;
  uniquePerSec: number;
  coverage: number; // max - min + 1 among uniques
  hit: boolean;
  indexToHit: number | null; // 1-based index to first hit if any
  estTimeToHitMs: number | null; // linear estimate using overall throughput
};

// ============================
// Core trial runner
// ============================

function runTrial(params: {
  N: bigint;
  mode: "mc_uniform" | "mc_phi" | "qmc" | "qmc_phi" | "stratified";
  samples: number;
  seed: number;
  phiAlpha: number; // curvature/bias exponent
  scramble: boolean; // Cranley–Patterson shift for QMC
}) {
  const { N, mode, samples, seed, phiAlpha, scramble } = params;
  if (samples <= 0) throw new Error("samples must be > 0");

  const rng = createXorShift32(seed);
  const sqrtN = isqrt(N);
  const start = performance.now();

  const uniques = new Set<string>();
  const values: bigint[] = [];
  let hit = false;
  let indexToHit: number | null = null;

  // prepare CP shift if needed
  const shift1 = scramble ? rng.next() : 0;
  const shift2 = scramble ? rng.next() : 0;

  for (let i = 0; i < samples; i++) {
    let u = 0;

    if (mode === "mc_uniform" || mode === "mc_phi") {
      u = rng.next();
    } else if (mode === "stratified") {
      // one stratum per sample
      const stratumStart = i / samples;
      const stratumEnd = (i + 1) / samples;
      u = stratumStart + rng.next() * (stratumEnd - stratumStart);
    } else if (mode === "qmc" || mode === "qmc_phi") {
      const [h1] = halton2(i);
      u = (h1 + shift1) % 1; // 1D CP shift
    }

    let cand: bigint;
    if (mode === "mc_phi" || mode === "qmc_phi") {
      cand = phiBiasedToCandidate(u, sqrtN, phiAlpha);
    } else {
      cand = linearMapToCandidate(u, sqrtN);
    }

    cand = clampCandidate(cand, sqrtN);
    values.push(cand);

    const key = cand.toString();
    if (!uniques.has(key)) uniques.add(key);

    if (!hit && cand >= 2n && N % cand === 0n) {
      hit = true;
      indexToHit = i + 1; // 1-based
      // continue to collect timing/coverage statistics
    }
  }

  const end = performance.now();
  const timeMs = end - start;
  const cps = (samples / Math.max(1e-9, timeMs)) * 1000;

  // coverage among uniques
  let min = values[0];
  let max = values[0];
  for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
  const coverage = Number(max - min + 1n);

  const unique = uniques.size;
  const uniquePerSec = (unique / Math.max(1e-9, timeMs)) * 1000;
  const estTimeToHitMs = indexToHit ? (timeMs / samples) * indexToHit : null;

  return {
    mode,
    samples,
    timeMs,
    cps,
    unique,
    uniquePerSec,
    coverage,
    hit,
    indexToHit,
    estTimeToHitMs,
    histogram: buildHistogram(values, sqrtN, 40),
    sqrtN,
  } as TrialMetrics & { histogram: { bin: string; count: number }[]; sqrtN: bigint };
}

// Human-friendly labels
const MODE_LABEL: Record<string, string> = {
  mc_uniform: "MC — Uniform",
  stratified: "MC — Stratified",
  mc_phi: "MC — φ‑biased",
  qmc: "QMC — Halton",
  qmc_phi: "QMC — Halton + φ‑bias",
};

// ============================
// React App
// ============================

export default function App() {
  const [nText, setNText] = useState<string>("899"); // default demo: 29×31
  const [samples, setSamples] = useState<number>(20000);
  const [seed, setSeed] = useState<number>(42);
  const [phiAlpha, setPhiAlpha] = useState<number>(PHI);
  const [scramble, setScramble] = useState<boolean>(true); // CP shift on by default
  const [selectedMode, setSelectedMode] = useState<"mc_uniform" | "stratified" | "mc_phi" | "qmc" | "qmc_phi">("qmc_phi");
  const [results, setResults] = useState<any[]>([]);

  const parsedN = useMemo(() => {
    try {
      const trimmed = nText.trim();
      if (!/^\d+$/.test(trimmed)) return null;
      const N = BigInt(trimmed);
      return N > 3n ? N : null;
    } catch {
      return null;
    }
  }, [nText]);

  const sqrtNStr = useMemo(() => {
    if (!parsedN) return "—";
    try {
      return isqrt(parsedN).toString();
    } catch {
      return "—";
    }
  }, [parsedN]);

  function runSelected() {
    if (!parsedN) return alert("Please enter a valid semiprime N (decimal, ≥ 4).");
    const m = runTrial({ N: parsedN, mode: selectedMode, samples, seed, phiAlpha, scramble });
    setResults((r) => [m, ...r]);
  }

  function runAll() {
    if (!parsedN) return alert("Please enter a valid semiprime N.");
    const modes: any[] = ["mc_uniform", "stratified", "mc_phi", "qmc", "qmc_phi"];
    const out = modes.map((mode) => runTrial({ N: parsedN, mode, samples, seed, phiAlpha, scramble }));
    setResults((r) => [...out, ...r]);
  }

  function reset() {
    setResults([]);
  }

  function copyJSON() {
    const payload = JSON.stringify(results, null, 2);
    navigator.clipboard.writeText(payload);
  }

  // Aggregate histogram for the most recent batch (first 5 items)
  const charts = results.slice(0, 5);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">QMC φ‑Biased RSA Candidate Sampler — Web Demo</h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={reset} title="Clear results"><RotateCcw className="w-4 h-4 mr-1"/>Reset</Button>
          <Button variant="outline" onClick={copyJSON} title="Copy latest results as JSON"><Copy className="w-4 h-4 mr-1"/>Copy JSON</Button>
        </div>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <Label htmlFor="n">Semiprime N (decimal)</Label>
            <Input id="n" value={nText} onChange={(e) => setNText(e.target.value)} placeholder="e.g., 899" />
            <p className="text-xs text-muted-foreground mt-1">√N ≈ {sqrtNStr}</p>
          </div>
          <div>
            <Label htmlFor="samples">Samples</Label>
            <Input id="samples" type="number" min={100} step={100} value={samples}
              onChange={(e) => setSamples(Math.max(100, Number(e.target.value) || 0))} />
          </div>
          <div>
            <Label htmlFor="seed">Seed</Label>
            <Input id="seed" type="number" value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)} />
          </div>
          <div>
            <Label>Mode</Label>
            <Select value={selectedMode} onValueChange={(v: any) => setSelectedMode(v)}>
              <SelectTrigger><SelectValue placeholder="Select mode"/></SelectTrigger>
              <SelectContent>
                <SelectItem value="mc_uniform">MC — Uniform</SelectItem>
                <SelectItem value="stratified">MC — Stratified</SelectItem>
                <SelectItem value="mc_phi">MC — φ‑biased</SelectItem>
                <SelectItem value="qmc">QMC — Halton</SelectItem>
                <SelectItem value="qmc_phi">QMC — Halton + φ‑bias</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="alpha">φ‑bias exponent α</Label>
            <Input id="alpha" type="number" step="0.05" value={phiAlpha}
              onChange={(e) => setPhiAlpha(Math.max(0.1, Number(e.target.value) || PHI))} />
            <p className="text-xs text-muted-foreground mt-1">Default α = φ ≈ {PHI.toFixed(6)} (skews toward √N)</p>
          </div>
          <div className="flex items-center gap-3 mt-6">
            <Switch id="scramble" checked={scramble} onCheckedChange={setScramble} />
            <Label htmlFor="scramble">Cranley–Patterson shift (QMC)</Label>
          </div>
        </CardContent>
      </Card>

      <div className="flex gap-2">
        <Button onClick={runSelected}><Play className="w-4 h-4 mr-1"/>Run Selected</Button>
        <Button variant="secondary" onClick={runAll}><Rocket className="w-4 h-4 mr-1"/>Run All Modes</Button>
      </div>

      <Tabs defaultValue="results">
        <TabsList>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="chart">Histogram</TabsTrigger>
          <TabsTrigger value="about">About</TabsTrigger>
        </TabsList>

        <TabsContent value="results">
          <Card>
            <CardHeader><CardTitle>Latest Trials</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left">
                    <tr className="border-b">
                      <th className="py-2 pr-4">Mode</th>
                      <th className="py-2 pr-4">Samples</th>
                      <th className="py-2 pr-4">Time (ms)</th>
                      <th className="py-2 pr-4">Cand/s</th>
                      <th className="py-2 pr-4">Unique</th>
                      <th className="py-2 pr-4">Unique/s</th>
                      <th className="py-2 pr-4">Coverage</th>
                      <th className="py-2 pr-4">Any Hit</th>
                      <th className="py-2 pr-4">Index→Hit</th>
                      <th className="py-2 pr-4">Est. ms→Hit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, idx) => (
                      <tr key={idx} className="border-b hover:bg-muted/40">
                        <td className="py-1 pr-4 font-medium">{MODE_LABEL[r.mode] ?? r.mode}</td>
                        <td className="py-1 pr-4">{r.samples}</td>
                        <td className="py-1 pr-4">{r.timeMs.toFixed(2)}</td>
                        <td className="py-1 pr-4">{r.cps.toFixed(0)}</td>
                        <td className="py-1 pr-4">{r.unique}</td>
                        <td className="py-1 pr-4">{r.uniquePerSec.toFixed(0)}</td>
                        <td className="py-1 pr-4">{r.coverage}</td>
                        <td className="py-1 pr-4">{r.hit ? "✓" : "—"}</td>
                        <td className="py-1 pr-4">{r.indexToHit ?? "—"}</td>
                        <td className="py-1 pr-4">{r.estTimeToHitMs ? r.estTimeToHitMs.toFixed(2) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chart">
          <Card>
            <CardHeader><CardTitle>Histogram (most recent runs)</CardTitle></CardHeader>
            <CardContent>
              {charts.length === 0 ? (
                <p className="text-sm text-muted-foreground">Run a trial to see distributions.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {charts.map((r, i) => (
                    <div key={i} className="h-64 border rounded-lg p-2">
                      <p className="text-xs mb-2 font-medium">{MODE_LABEL[r.mode] ?? r.mode}</p>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={r.histogram}>
                          <XAxis dataKey="bin" hide/>
                          <YAxis hide/>
                          <Tooltip/>
                          <Bar dataKey="count" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="about">
          <Card>
            <CardHeader><CardTitle>About this Demo</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm leading-relaxed">
              <p>
                This in‑browser demo compares <b>Monte Carlo (MC)</b> and <b>Quasi‑Monte Carlo (QMC)</b> sampling
                schemes for generating integer candidates in the interval [2, √N] for trial division against a semiprime N.
                QMC uses a 1‑D Halton sequence with optional Cranley–Patterson random shift; φ‑bias skews the sampling toward √N
                using the exponent α (default α = φ) so that QMC+φ emphasizes regions that often contain factors for balanced semiprimes.
              </p>
              <ul className="list-disc pl-6">
                <li><b>Candidates/s</b> and <b>Unique/s</b> show throughput and redundancy reduction.</li>
                <li><b>Coverage</b> is (max−min+1) among sampled candidates.</li>
                <li><b>Any Hit</b> indicates whether any sampled candidate divides N; <b>Index→Hit</b> is the 1‑based index of the first hit, with a linear estimate of ms→hit.</li>
                <li>For very large N (≫ 1e15), some UI mappings rely on Number casts and are for demonstration only.</li>
              </ul>
              <p>
                Suggested usage: run all modes on N=899 (29×31) to reproduce the qualitative behavior reported in your PR; then toggle scrambling,
                vary α, and compare MC vs QMC under identical transforms.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <footer className="text-xs text-muted-foreground">
        Built for an educational demonstration of variance reduction in RSA candidate generation. For rigorous claims, run repeated trials and bootstrap CIs.
      </footer>
    </div>
  );
}
