# SlideVQA Evaluation: AgentLoop vs aquery(vlm_enhanced)

Date: 2026-04-29  
Dataset shard: `slidevqa_test/data/test-00001-of-00012.parquet`  
QA groups: `slidevqa_test/cache/qa_groups/test-00001.json`

## Runs Compared

| Method | Run ID | Decks | QA judged | Total score | Average |
|---|---:|---:|---:|---:|---:|
| AgentLoop + retrieve/image tools | `test_00001_full` | 40/40 | 185 | 15470 | 83.62 |
| Direct `RAGAnything.aquery(..., vlm_enhanced=True)` | `test_00001_aquery_vlm` | 40/40 | 185 | 15796 | 85.38 |

Net result:

```text
aquery improvement: +326 total points over 185 QA
average improvement: +1.76 / 100
```

Both runs reused the same parsed deck cache and RAG storage as much as possible. The aquery run changed the answering path only: it replaced the AgentLoop workflow with direct `engine.aquery(question, mode="hybrid", vlm_enhanced=True)`.

## Score Distribution

| Score bucket | AgentLoop | aquery VLM |
|---|---:|---:|
| 90-100 | 150 | 155 |
| 80-89 | 2 | 4 |
| 60-79 | 0 | 0 |
| 40-59 | 9 | 2 |
| 1-39 | 0 | 1 |
| 0 | 24 | 23 |

Question-level comparison:

```text
aquery better: 27 questions
AgentLoop better: 18 questions
Tie: 140 questions
AgentLoop <90: 35 questions
aquery <90: 30 questions
AgentLoop 0-score: 24 questions
aquery 0-score: 23 questions
```

The gain is real but moderate. aquery is not universally better; it improves several high-impact misses, but introduces its own retrieval/context-selection errors.

## Biggest Deck-Level Improvements

| Deck | AgentLoop avg | aquery avg | Delta |
|---|---:|---:|---:|
| `cmw2014workshopv2-140912093008-phpapp01_95` | 50.00 | 91.67 | +41.67 |
| `apachekafkaatlinkedin-150108003805-conversion-gate01_95` | 60.00 | 100.00 | +40.00 |
| `beerindustryfinalslides-110814110012-phpapp01_95` | 50.00 | 81.67 | +31.67 |
| `amitpresentation-141220101151-conversion-gate01_95` | 83.33 | 100.00 | +16.67 |
| `buildingmicroserviceswithscalafunctionaldomainmodelsandspringbootchrisrichardson-141030115818-conversion-gate01_95` | 83.33 | 100.00 | +16.67 |
| `cleanhydrocarbonsrenewableelectrificationsolarenergyinsouthernalberta-160323184734_95` | 85.00 | 98.57 | +13.57 |
| `antibiotics-141109111447-conversion-gate02_95` | 84.29 | 97.14 | +12.86 |
| `consultselling-150818132925-lva1-app6892_95` | 86.25 | 97.50 | +11.25 |

Observed pattern: aquery helps most when the answer is present in image/table evidence and the AgentLoop either failed to select the right evidence or answered too generically.

## Biggest Deck-Level Regressions

| Deck | AgentLoop avg | aquery avg | Delta |
|---|---:|---:|---:|
| `colors-141027065047-conversion-gate01_95` | 90.00 | 56.67 | -33.33 |
| `auto-100113133616-phpapp01_95` | 99.29 | 70.71 | -28.57 |
| `b2b-social-testing-lopez-141009140744-conversion-gate02_95` | 100.00 | 71.43 | -28.57 |
| `architecture-150102195159-conversion-gate01_95` | 100.00 | 73.75 | -26.25 |
| `astudyoncustomerpreferebceandsatisfactiontowardsbajajbikes-150814133516-lva1-app6891_95` | 100.00 | 85.71 | -14.29 |
| `bigdataplatformatpinterest-awsloft-150918193740-lva1-app6891_95` | 12.50 | 0.25 | -12.25 |

Observed pattern: aquery can pick the wrong nearby chart, wrong page, or wrong entity when the retrieval context contains several visually similar candidates. AgentLoop sometimes does better because it synthesizes from retrieved text and tool outputs more conservatively.

## Representative aquery Wins

1. `apachekafkaatlinkedin`, QA 285  
   Question: Where is Sam Shah based?  
   Ground truth: `Mountain View, California`  
   AgentLoop: no evidence found, score 0  
   aquery: `Sam Shah is based in Mountain View, California`, score 100

2. `apachekafkaatlinkedin`, QA 286  
   Question: What is Albert Wang's position?  
   Ground truth: `Senior User Experience Designer`  
   AgentLoop: no evidence found, score 0  
   aquery: `Senior User Experience Designer at LinkedIn`, score 100

3. `amitpresentation`, QA 208  
   Question: Utilities share in Apple App Store categories?  
   Ground truth: `5`  
   AgentLoop: `7%`, score 0  
   aquery: `5%`, score 100

4. `beerindustryfinalslides`, QA 326  
   Question: Top four brewers with 2.6% sales increase in 2010?  
   Ground truth: `A-B InBev, SABMiller, Heineken, Carlsberg`  
   AgentLoop: said deck does not specify, score 0  
   aquery: answered all four, score 100

5. `ch16`, QA 190  
   Question: What are four HR Deliverables?  
   Ground truth: `Employment stability, Team-based behaviors, Strategy-focused behaviors, High-talent staffing level`  
   AgentLoop: unrelated HR items, score 0  
   aquery: score 100

6. `cleanhydrocarbons...`, QA 226  
   Question: author of Clean Hydrocarbons & Renewable Electrification?  
   Ground truth: `Krzysztof Palka`  
   AgentLoop: `Imaginea Energy`, score 0  
   aquery: score 100

## Representative aquery Regressions

1. `architecture`, QA 349  
   Question: Where does Horizon send the HTTP request to?  
   Ground truth: `Keystone`  
   AgentLoop: `Keystone`, score 100  
   aquery: `Nova API`, score 0

2. `auto`, QA 308  
   Question: Do more used or new car buyers go online to have fun?  
   Ground truth: `New Car Buyers`  
   AgentLoop: correct, score 100  
   aquery: reversed the comparison, score 0

3. `b2b-social-testing`, QA 247  
   Question: Twitter username for the Moz CEO?  
   Ground truth: `SarahBird`  
   AgentLoop: correct, score 100  
   aquery: insufficient information, score 0

4. `b2b-social-testing`, QA 249  
   Question: Which day did the most Facebook fans of Moz see their posts?  
   Ground truth: `THU`  
   AgentLoop: correct, score 100  
   aquery: `Wednesday`, score 0

5. `colors`, QA 338  
   Question: color whose split complementary colors are red and green?  
   Ground truth: `NOBILITY, ROYALTY, LUXURY, AMBITION`  
   AgentLoop: score 95  
   aquery: score 0

## Persistent Failure Case

`americalooksto2024keyfindings-140701054257-phpapp01_95` remains a hard case for both methods.

| QA | Ground truth | AgentLoop | aquery |
|---|---:|---:|---:|
| 257 | 31 percentage points | 0 | 0 |
| 258 | 31 percentage points | 0 | 0 |
| 259 | Less | 0 | 0 |

This deck asks about a chart on hard work / playing by rules in 10 years. The correct calculation is:

```text
10 years ago more likely: 61%
in 10 years more likely: 30%
drop: 61 - 30 = 31 percentage points
```

Both methods retrieve or emphasize visually similar but wrong percentages. This suggests the remaining issue is not just AgentLoop vs aquery; it is evidence selection and page-level visual verification for chart arithmetic.

## Interpretation

aquery with `vlm_enhanced=True` improves the final score from 83.62 to 85.38, a +1.76 point gain. The main reasons are:

1. Direct VLM-enhanced answering can use image chunks more effectively.
2. It avoids some AgentLoop planning/tool-use failures where the agent says evidence is missing despite useful evidence being indexed.
3. It often gives more precise short answers when the relevant visual crop is retrieved.

However, aquery also regresses on some decks because:

1. It can choose the wrong visual candidate from a dense retrieved prompt.
2. It has less explicit multi-step reasoning/control than an agent loop.
3. It can over-trust retrieved context and answer from a similar but incorrect page/chart.

## Recommendation

Do not replace AgentLoop with aquery wholesale based only on this result. The best next approach is likely a hybrid:

1. Keep AgentLoop for control and multi-step reasoning.
2. Add a targeted `inspect_page(page_number, prompt)` or equivalent VLM page inspection tool.
3. Preserve page metadata in retrieve outputs: `page_idx`, `source_page`, `image_path`, `bbox`.
4. For chart/table/percentage/comparison/arithmetic questions, require second-pass visual verification on the most relevant page.
5. Use aquery-vlm as a fallback or secondary answer generator, then compare against AgentLoop answer when confidence is low.

Expected benefit: retain AgentLoop's wins on structured/reasoning cases while capturing aquery's gains on visual evidence and small text.

