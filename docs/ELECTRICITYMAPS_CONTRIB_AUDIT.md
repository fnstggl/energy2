# Electricity Maps Contrib Audit + Aurelius Market-Data Integration Plan

Audited repository: <https://github.com/electricitymaps/electricitymaps-contrib>
Audit date: 2026-05-22
Audit method: full shallow clone into a throwaway external directory
(`/tmp/electricitymaps-contrib`, **not** committed to Aurelius), inspection of
`config/zones/*.yaml`, `config/data_centers/data_centers.json`,
`electricitymap/contrib/parsers/*.py`, `DATA_SOURCES.md`, and the license files.

> **Could the repo be fully inspected?** Yes. The repository was cloned and
> inspected in full (zone configs, parsers, data-center mappings, data-source
> docs, and license files). Findings below are first-hand, not assumed.

---

## 1. Executive summary

Electricity Maps contrib is a **carbon-intensity and electricity-mix** project,
not a wholesale-price project. For Aurelius it is most valuable as a
**reference** for:

1. **Official data-source identification** (`DATA_SOURCES.md` lists the
   authoritative operator/TSO behind every zone).
2. **Zone naming + geography** (`config/zones/`, `config/zone_names.json`).
3. **Cloud-region → grid-zone geography** (`config/data_centers/data_centers.json`
   maps 147 AWS/Azure/GCP/OVH data-center regions to grid zones).
4. **Normalization patterns** (timezone handling, per-mode production
   normalization, estimation flags).

It is **not** a usable source of wholesale prices for the US ISOs Aurelius
targets, and its EU prices come from ENTSO-E — the same source Aurelius already
reads directly. The recommended posture is therefore unchanged from Aurelius'
existing design: **direct ISO/TSO is source-of-truth; Electricity Maps is an
optional carbon aggregator / sandbox / fallback only.**

---

## 2. What the repo provides

| Asset | Path | Use to Aurelius |
| --- | --- | --- |
| Zone definitions (capacity, bounding box, parser bindings, sources) | `config/zones/*.yaml` | Reference for zone keys + official sources |
| Zone display names | `config/zone_names.json` | Reference only |
| Cloud data-center → zone map | `config/data_centers/data_centers.json` | **High value**: cloud-region aliases for the region registry |
| Parsers (carbon/production/consumption/exchange/price) | `electricitymap/contrib/parsers/*.py` | Reference for endpoints + normalization, **not** for copying (AGPL) |
| Authoritative source list | `DATA_SOURCES.md` | Reference for the operator-of-record per country/zone |
| Emission factors sourcing | `EMISSION_FACTORS_SOURCES.md` | Reference if/when Aurelius computes its own intensities |

### Signals that ARE present and useful
- **Carbon intensity** (gCO₂eq/kWh) — historical + real-time, all mapped zones.
- **Electricity mix / production breakdown** by mode (coal, gas, wind, solar…).
- **Consumption / load** (often via EIA for US zones).
- **Interconnector / exchange flows** between zones.
- **Renewable percentage** (derivable from the production breakdown).
- **Day-ahead price** — **only for ENTSO-E and a handful of non-US zones**
  (see §3).

---

## 3. What the repo does NOT provide (critical)

1. **No US ISO wholesale prices.** Inspected the `parsers:` bindings of every US
   zone (`US-CAL-CISO`, `US-MIDA-PJM`, `US-TEX-ERCO`, `US-NY-NYIS`,
   `US-MIDW-MISO`, `US-CENT-SWPP`, `US-NE-ISNE`). None bind a `price:` parser —
   they bind only `production`, `consumption`, `*Forecast`, `exchange`. So for
   CAISO/PJM/ERCOT/NYISO/MISO/SPP/ISO-NE the contrib repo yields **carbon and
   generation mix only — no LMP, no day-ahead, no real-time price.**
2. **No nodal LMP anywhere.** 84 zones bind a `price:` parser, but they are all
   European (`ENTSOE.py`) or other international operators (`CA_AB`, `CA_ON`,
   `GB`, `JP`, `CO`, `CAMMESA`, …). Every one of these returns a **zonal /
   bidding-zone / country-level day-ahead price**, never a true nodal LMP.
3. **EU prices are just ENTSO-E re-reads.** `ENTSOE.fetch_price()` queries the
   ENTSO-E Transparency Platform (`documentType=A44`). Aurelius already reads
   ENTSO-E directly (`grid_apis/entsoe.py`), so the contrib parser adds nothing
   but an extra hop.
4. **It is not a price-of-record for settlement.** Even where prices exist, they
   are re-published for display, not for financial settlement.

---

## 4. Which parts help Aurelius

| Need | Use contrib for? | How |
| --- | --- | --- |
| US wholesale prices | ❌ No | Use direct ISO/TSO (already implemented) |
| EU day-ahead prices | ⚠️ Reference only | Read ENTSO-E directly (already implemented) |
| Carbon intensity | ✅ As optional aggregator/fallback | Electricity Maps **API** (not the repo) |
| Zone keys / names | ✅ Reference | Copied factually into `region_registry.py` |
| Cloud-region → grid mapping | ✅ Reference (high value) | Verified subset adapted into `region_registry.py` |
| Official source identification | ✅ Reference | Confirms our ISO endpoints |
| Normalization patterns | ✅ Concept only | Clean-room reimplementation |

---

## 5. Legal / licensing notes

- The repo is **GNU AGPL-3.0** since v1.5.0 (`LICENSE.md`). Contributions before
  commit `cb9664f` were MIT (`LICENSE_MIT.txt`), but you cannot assume any given
  current file is the old MIT version — treat the repository as **AGPL**.
- **AGPL is copyleft with a network clause.** Copying substantial parser code
  into Aurelius would risk obligating Aurelius (including networked use) under
  AGPL. **We did not copy any parser code.**
- **What we did use, and why it is safe:**
  - **Factual identifiers** — zone keys (`US-CAL-CISO`), operator names, ENTSO-E
    EIC domain codes, official source URLs. These are facts, not creative
    expression, and are not copyrightable.
  - **Cloud-region geography** — a small, hand-verified subset of
    `data_centers.json` (region → grid-zone), re-expressed in our own schema in
    `region_registry.py`. Factual geography, not copied code.
  - **Concepts** — "flag estimated values", "normalize to UTC", "one canonical
    zone key" — reimplemented clean-room.
- **Bright line:** no AGPL source files were vendored, imported, or
  copy-pasted. The clone lived only in `/tmp` and is **not** committed.

---

## 6. Useful zones / sources table

Canonical Aurelius region → operator (source of truth) → Electricity Maps zone
→ price availability. Mirrors `aurelius/ingestion/region_registry.py`.

| Aurelius region | ISO/TSO (source of truth) | Source region/hub | EM zone | EM price? | Nodal LMP? | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| us-west | CAISO (OASIS, no auth) | TH_NP15_GEN-APND | US-CAL-CISO | ❌ | ✅ | high |
| us-east | PJM (Data Miner, key) | PJM-RTO pnode 1 | US-MIDA-PJM | ❌ | ✅ | high |
| us-south | ERCOT (Public API, creds) | HB_HOUSTON | US-TEX-ERCO | ❌ | ❌ (SPP, not LMP) | high |
| us-central | SPP (not implemented) | SPP-system | US-CENT-SWPP | ❌ | ✅ | low |
| us-north | MISO (not implemented) | MISO-system | US-MIDW-MISO | ❌ | ✅ | low |
| us-newengland | ISO-NE (not implemented) | ISONE-system | US-NE-ISNE | ❌ | ✅ | low |
| us-nyiso | NYISO (not implemented) | NYISO-zones | US-NY-NYIS | ❌ | ✅ | low |
| eu-west | ENTSO-E (DE/EnBW) | 10YDE-ENBW-----N | DE | ✅ (=ENTSO-E) | ❌ (bidding zone) | medium |
| eu-north | ENTSO-E (NO1/Statnett) | 10YNO-1--------2 | NO-NO1 | ✅ (=ENTSO-E) | ❌ (bidding zone) | medium |
| eu-central | ENTSO-E (FR/RTE) | 10YFR-RTE------C | FR | ✅ (=ENTSO-E) | ❌ (bidding zone) | medium |

Cloud-region aliases (verified subset from `data_centers.json`):
`aws us-west-1`, `gcp us-west2`, `azure westus` → **us-west**;
`aws us-east-1/us-east-2`, `gcp us-east4/us-east5`, `azure eastus/eastus2/northcentralus` → **us-east**;
`gcp us-south1`, `azure southcentralus` → **us-south**;
`gcp us-central1` → **us-central** (SWPP); `azure centralus` → **us-north** (MISO).
No AWS region maps to ERCOT, MISO, NYISO, or ISO-NE in the contrib data set.

---

## 7. Suggested provider architecture

```
                    +-------------------------------+
                    |   region_registry.py          |  canonical region ->
                    |   (ISO + EM zone + cloud)     |  ISO/EM/cloud + confidence
                    +---------------+---------------+
                                    |
        +---------------------------+----------------------------+
        |                           |                            |
+---------------+        +----------------------+      +-------------------+
| Direct ISO/TSO|        | Electricity Maps API |      | CSV / fixtures     |
| (source of    |        | (carbon aggregator,  |      | (offline/tests)    |
|  truth)       |        |  sandbox, fallback)  |      |                    |
| CAISO/PJM/    |        | carbon only;         |      |                    |
| ERCOT/ENTSO-E |        | NO US prices, NO LMP |      |                    |
+-------+-------+        +----------+-----------+      +---------+---------+
        |                           |                            |
        +-----------+---------------+----------------------------+
                    |
        +-----------v-------------------------------+
        | market_data_provider.py                   |
        |  MarketDataProvider (capabilities,        |
        |  price/carbon series), typed points with  |
        |  provenance + is_sandbox + is_estimated   |
        +-----------+-------------------------------+
                    |
        +-----------v---------------+      +---------------------------+
        | points_to_price_df /      |----->| grid_apis/base.py         |
        | points_to_carbon_df       |      | canonical DataFrame schema|
        +---------------------------+      +---------------------------+
                    |
        +-----------v---------------+
        | assert_benchmark_admissible  -> blocks sandbox/randomized data
        | from savings/benchmark claims
        +---------------------------+
```

Routing rule: **prefer the highest-confidence source-of-truth provider whose
capabilities cover the region+signal; fall back to the aggregator only for
carbon, and never let sandbox/estimated data reach a savings claim.**

---

## 8. Risks of depending on the Electricity Maps API

1. **Single-aggregator lock-in.** If Aurelius routed prices/carbon solely
   through Electricity Maps, an outage, pricing change, or zone redefinition on
   their side would silently degrade Aurelius. Mitigation: provider abstraction
   + direct ISO/TSO as primary.
2. **No US prices / no LMP.** Relying on EM for prices is impossible for US ISOs
   and wrong (zonal vs nodal) elsewhere.
3. **Estimation.** EM marks some values estimated/modelled. Treating estimated
   carbon as measured would overstate precision — hence the `is_estimated`
   flag.
4. **Sandbox confusion.** EM offers sandbox/demo data that is randomized. Using
   it for savings claims would be fraudulent-by-accident — hence the hard
   `is_sandbox` gate.
5. **Rate limits / paid tiers.** Free tier ~30 req/min, ~90-day history. History
   depth and SLA require a paid plan.

---

## 9. How to use the Electricity Maps sandbox safely

- Set `ELECTRICITYMAPS_SANDBOX=true` to exercise connectors/schemas without a
  production key.
- The provider flags **every** sandbox observation `is_sandbox=True` /
  `provenance=sandbox`.
- `assert_benchmark_admissible()` **raises** if any sandbox point reaches a
  benchmark/savings path. `filter_benchmark_admissible()` drops them.
- **Sandbox/randomized data is for connector and schema tests only.**
- **Production benchmark claims require real, unrandomized historical data**
  from the source-of-truth ISO/TSO (or a paid EM plan for carbon).
- The API token is never logged or printed; `repr()` shows only `<set>`/`<unset>`.

---

## 10. Direct ISO/TSO source recommendations (source of truth)

| Market | Endpoint | Auth | Notes |
| --- | --- | --- | --- |
| CAISO | OASIS `SingleZip` PRC_LMP (DAM) / PRC_INTVL_LMP (RTM) | none | NP15 hub, ZIP/CSV, ~31-day window |
| PJM | Data Miner `da_hrl_lmps` / `rt_fivemin_hrl_lmps` | PJM_API_KEY | RTO aggregate pnode 1 |
| ERCOT | Public API `dam_stlmnt_pnt_prices` / `spp_node_zone_hub` | ERCOT creds | HB_HOUSTON SPP (not LMP) |
| ENTSO-E | Transparency `documentType=A44` | ENTSOE_API_KEY | Bidding-zone day-ahead (EUR/MWh) |
| MISO/SPP/NYISO/ISO-NE | operator marketplace portals | registration | **Not yet implemented** — confidence=low in registry |
| Carbon (any) | Electricity Maps API `/v3/carbon-intensity/past-range` | ELECTRICITYMAPS_API_KEY | aggregator/fallback; WattTime for marginal MOER |

---

## 11. Exact next implementation steps

1. **Route all EM zone lookups through `region_registry.py`** (done for the new
   provider; migrate any remaining hard-coded maps in `cli.py`).
2. **Wire `MarketDataProvider` capabilities into source selection** so the
   optimizer/backtester asks the registry for the best source-of-truth provider
   per region+signal before falling back to EM.
3. **Call `assert_benchmark_admissible()` inside the savings/benchmark harness**
   (`benchmarks/`, `aurelius/reporting/savings_report.py`) so sandbox/estimated
   data can never back a savings number.
4. **Implement SPP/MISO/NYISO/ISO-NE price providers** to raise their registry
   confidence from `low`; until then they remain carbon-only.
5. **Add a paid-tier EM history note** to `.env.example` and obtain real
   historical carbon for any carbon-savings claims (sandbox is test-only).
6. **Optional:** adapt more of `data_centers.json` (clean-room) if Aurelius adds
   workload region pinning for more cloud regions.
