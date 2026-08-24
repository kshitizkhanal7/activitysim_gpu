# ChoiceForge: A Plain-English Guide to Faster Travel Demand Modeling with GPUs

## Who this guide is for

This guide is for a curious high school student. You do not need to know transportation planning, probability, programming, or computer hardware. By the end, you should understand:

- what a travel demand model tries to predict;
- how a computer turns possible choices into probabilities;
- why the same calculation must be repeated millions of times;
- what CPUs, GPUs, and GPU kernels do;
- what ChoiceForge changes;
- what the early benchmark results do and do not prove; and
- why faster modeling could matter to communities.

## The one-minute version

A city may want to know what could happen if it adds a bus line, changes a toll, builds housing, or closes a bridge. It cannot test every idea in the real world first, so planners use a **travel demand model**: a computer simulation of how people may decide where, when, why, and how to travel.

Modern models can represent millions of people and many possible choices. For each person, the computer scores the available choices, turns the scores into probabilities, and uses a controlled random number to select one outcome. It repeats this work again and again for trips, tours, destinations, travel modes, and times of day.

A normal processor, called a **CPU**, is very flexible but has a modest number of powerful processing cores. A **GPU** has many more, simpler cores and is good at doing the same kind of arithmetic for a huge number of records at once. A **GPU kernel** is a small program designed to run across those GPU cores.

ChoiceForge is an open-source experiment that moves a particularly repetitive travel-choice calculation to an NVIDIA GPU. Its most important idea is **fusion**: calculate utility scores, apply availability rules, compute a logsum, and make the choice in one GPU operation, without first storing an enormous table of intermediate scores.

The first synthetic test compared the GPU with a readable NumPy reference. Phase 1 added a much stronger 24-core CPU competitor. Phase 2 captured 4,477 real mandatory-tour scheduling decisions from ActivitySim, and every GPU choice matched. It also found a problem: transferring a 151.6-megabyte table made the GPU slower. Phase 3 replaced that table with 22.0 megabytes of compact ingredients and made the captured calculation 3.83 times faster than a purpose-built 48-thread CPU version. Phases 4 and 5 installed and reused the GPU path in real scheduling. Phase 6 handled real destination work. Phase 7 batched repeated setup and added nested logit on the GPU. Phase 8 moved the work to pinned current ActivitySim and a public 50,000-household run. Phase 9 then used the much larger public MTC geography. It confirmed an exact GPU destination result and also caught a scheduling mismatch before that path could be claimed as correct. Phase 11 repeated the full-geography destination experiment three times and established the supported production result: the complete 50,000-household model was 1.064 times faster with byte-for-byte identical outputs. Phase 12 built the foundation for moving the large utility equations themselves onto the GPU. Phase 13 completed the strict CPU answer key. Phase 14 generated GPU code from that same recipe and matched the CPU exactly on 30 real public-model batches. Phase 15 connected it to the real model, removed the giant intermediate table, proved a small destination-component win, and found an honest scale limit. Phase 16 added compact inputs, caches, and a published FP32 policy. Its CPU and GPU matched every checked cell, and three large public pairs proved a repeated 1.025-times trip-destination component speedup with exact modeled decisions. Phase 17 made the compiled GPU work reusable and continued it into trip mode choice. Five large public pairs improved trip destination by 1.040 times with exact modeled decisions; the complete-model median also improved by 1.006 times, although its strict statistical gate still did not pass.

Phase 18 then kept a dependent model-shaped chain on the GPU and processed all
2.875 million public households, but its behavior equations were synthetic.
Phase 19 replaced that synthetic boundary with the public calibrated
auto-ownership and mandatory-tour-frequency equations. It exactly reproduced
50,000 saved household choices and 78,900 saved person choices. Median GPU
modeled work was 17.840 times faster than the independent CPU replay and 12.073
times faster after including one input upload and final download. That is the
strongest current calibrated result, although upstream location/CDAP work and
downstream tour creation are still outside its boundary.

## 1. What is travel demand modeling?

Imagine a region with homes, schools, offices, stores, roads, sidewalks, rail lines, and buses. Transportation planners ask questions such as:

- How many people might travel during the morning rush?
- Where might they go?
- Will they drive, walk, bike, take transit, or share a ride?
- Which roads or transit services might become crowded?
- Who benefits from a project, and who may be left out?
- How might results change if fuel prices, fares, jobs, or housing change?

A travel demand model does not read minds or predict one person's future with certainty. It creates a plausible regional picture by applying observed patterns and mathematical rules to many simulated people.

### People, households, and zones

Models usually begin with a **synthetic population**. These are made-up records that resemble the region's real population without being a list of actual named residents. A record may describe a household's size, income, vehicles, workers, and students.

The region is divided into geographic areas called **zones**. A zone may contain homes, jobs, schools, shops, or transit stations. The model also needs information about travel between zones, such as distance, driving time, transit time, cost, and whether a route is available. These travel measurements are often called **skims**.

### Activities, tours, and trips

People travel because they want to do activities. Going from home to school is a trip. Going from home to school, then to a store, then home is often treated as a **tour**: a chain of trips that begins and ends at the same anchor, usually home or work.

An activity-based model may answer a sequence of linked questions:

```text
Who is traveling?
    -> What activity are they doing?
    -> Where will they go?
    -> What time will they travel?
    -> Which travel mode will they use?
    -> Which trips and tours result?
```

These decisions affect one another. A faraway destination may make walking unrealistic. A late return time may make transit unavailable. A household with one vehicle and two workers may face competition for the car.

## 2. How does a model represent a choice?

Suppose Maya must choose among walking, taking the bus, and riding in a car. The model gives every option a **utility score**. Utility is not electricity here. It is a numerical score for how attractive an option is according to the model.

A simplified utility equation looks like this:

```text
utility = constant + (time x time weight) + (cost x cost weight) + other terms
```

A negative time weight means longer trips are less attractive. A negative cost weight means expensive trips are less attractive. The weights are estimated from travel surveys or other observations. Different people can receive different scores because income, age, vehicle access, location, and other facts may matter.

### Availability comes first

Not every option is possible. Walking may be unavailable for a very long trip. A transit route may not exist. A person may not have access to a car. The model applies an **availability rule** so an impossible option receives no chance of being chosen.

### From scores to probabilities

Raw utility scores are not probabilities. A common model uses this rule:

```text
probability of option i = exp(utility i) / sum of exp(utility for every available option)
```

`exp` is a mathematical function that turns higher scores into much larger positive numbers. Dividing by the total makes all probabilities add to 1, or 100 percent.

Here is a small example:

| Option | Utility score | exp(score) | Probability |
|---|---:|---:|---:|
| Walk | 0 | 1.000 | 9.0% |
| Bus | 1 | 2.718 | 24.5% |
| Car | 2 | 7.389 | 66.5% |

The car is most likely, but it is not guaranteed. This matters because real people with similar situations do not all make the same decision.

### Turning probabilities into one choice

The model takes a random draw between 0 and 1. It lays the probabilities end to end:

```text
Walk: 0.000 to 0.090
Bus:  0.090 to 0.335
Car:  0.335 to 1.000
```

If the draw is 0.600, the model chooses car. If it is 0.200, it chooses bus. This is called **inverse cumulative distribution sampling**, but the important idea is simple: larger probability ranges are more likely to catch the draw.

ChoiceForge does not invent random numbers inside the GPU. The calling model supplies them. That design lets the CPU and GPU use the exact same draws, which makes fair comparison possible and preserves ActivitySim's rules for repeatable simulations.

### What is a logsum?

A **logsum** summarizes how good the full set of choices is:

```text
logsum = log(sum of exp(utility for every available option))
```

In the example, the logsum is about 2.407. If a fast, affordable new bus service improves one option, the logsum rises. Models often pass this accessibility-like value into later decisions.

Computers must calculate logsums carefully. Very large utility scores can make `exp(score)` overflow, like a calculator displaying an error for a number that is too large. The standard safe method subtracts the largest utility before exponentiating and adds it back afterward. This is called **stable logsum-exp**. ChoiceForge uses this method.

## 3. Why does this become a computing problem?

The three-option example is tiny. A regional model may have:

- millions of people, households, tours, or trips;
- dozens of alternatives for mode or time choices;
- hundreds or thousands of possible destinations;
- many variables in every utility equation; and
- dozens of model steps that repeat similar calculations.

If one million choosers each consider 32 alternatives, the model evaluates 32 million chooser-option combinations in just one component. Destination choice can be much larger.

A straightforward program may create a rectangular **utility matrix** with one row per chooser and one column per alternative. For one million choosers and 32 alternatives stored as 32-bit numbers, that table alone uses 128 megabytes. The program may then create more tables for exponentials, probabilities, or cumulative probabilities. Moving and storing these intermediate values can take as much time as the arithmetic.

## 4. CPU, GPU, and kernel in everyday language

A **CPU** is the general-purpose brain of a computer. Think of it as a small team of highly trained chefs. Each chef can handle complicated instructions, switch tasks quickly, and make decisions.

A **GPU** was designed to calculate many screen pixels at once. Think of it as a very large kitchen crew. Each worker is less independent, but thousands of workers can perform the same short recipe on different ingredients at the same time.

A **GPU kernel** is that short recipe. It says what one group of GPU threads should calculate. Launching a kernel sends many copies of the recipe across the data.

The GPU is not automatically faster. Three costs matter:

1. **Transfer:** data may need to cross from ordinary computer memory to GPU memory and back.
2. **Launch overhead:** starting a GPU kernel takes a small amount of time.
3. **Parallel work:** the job must be large and regular enough to keep many GPU cores busy.

Small or irregular jobs may be faster on the CPU. A good GPU design must move enough connected work together and avoid unnecessary transfers.

## 5. The main ChoiceForge idea: fuse the pipeline

A less efficient pipeline might do this:

```text
CPU creates inputs
    -> GPU computes and stores every utility
    -> another operation reads utilities and computes probabilities
    -> another operation reads probabilities and selects choices
    -> results return to CPU
```

ChoiceForge's fused linear kernel aims to do this:

```text
chooser features + alternative weights + constants + availability + random draw
    -> one fused GPU kernel
    -> selected alternative + logsum
```

Inside the kernel, one GPU block handles one chooser. Threads in the block work on the alternatives together. They calculate utility scores, ignore unavailable options, find the maximum score, form safe exponential weights, sum them, select the option containing the supplied random draw, and return only the useful outputs.

The intermediate chooser-by-alternative utility matrix is never written to large global GPU memory. Temporary values stay close to the GPU cores in faster **shared memory**. This reduces memory traffic and storage.

The milestone-one kernel supports:

- a fixed set of 1 to 1,024 alternatives;
- linear utility equations using 32-bit floating-point numbers;
- availability rules;
- stable logsums;
- externally supplied random draws; and
- exact comparison of selected alternatives with a CPU reference.

It does not yet support all of ActivitySim's expression language, estimation mode, or destination sets with thousands of alternatives. Phase 7 supports the canonical MTC 21-mode nested-logit tree, but not every irregular ActivitySim choice model.

## 6. How the prototype is kept honest

Speed is useless if the model silently changes its answers. The project therefore starts with a readable NumPy CPU implementation called the **reference** or **oracle**. The GPU result is checked against it.

The validation rules are:

- give the CPU and GPU identical inputs and identical random draws;
- preserve the same alternative order and boundary rules;
- require every selected alternative to match exactly;
- measure the numerical difference between CPU and GPU logsums;
- test unavailable alternatives and invalid rows; and
- report mismatches instead of hiding them.

Small floating-point differences are normal because parallel GPU operations may add numbers in a different order. It is like adding a long list of rounded decimals from left to right versus pairing them first: the last digit can differ. The selected choices still must match, and logsum differences must stay within an explicit tolerance.

The Python 3.11 ActivitySim and CUDA integration environment now passes 95 tests. These include exact comparison with ActivitySim's real Numba choice function, compact expression compilation, segmented destination batches, categorical flags, nested-logit validation, current-version fallback forwarding, real scheduling integration, GPU tests using 33 and 190 alternatives, a safe expression interpreter, skim-table adapters, shadow checks, the strict CPU reference, the generated strict CUDA evaluator, the explicit FP32 policy used for Phase 16, Phase 17's schema-safe plan and workspace reuse, and Phase 18's fail-closed GPU state, stable random draws, deterministic partitions, and ordered aggregation.

## 7. What was benchmarked?

A **benchmark** is a controlled timing experiment. ChoiceForge's first benchmark uses synthetic, meaning computer-generated, data. Each test has:

- 32 alternatives;
- 16 features per chooser;
- 10,000, 100,000, or 1,000,000 choosers; and
- 32-bit floating-point calculations.

It compares three timings:

1. **NumPy materialized:** the readable CPU reference creates the utility matrix.
2. **GPU including transfers:** inputs move to the GPU, the kernel runs, and outputs move back.
3. **GPU resident:** needed arrays are assumed to already be in GPU memory.

The GPU-resident number shows the potential of a future pipeline that keeps related model steps on the GPU. The transfer-inclusive number is the more conservative measure for a single isolated call.

All numbers below are medians after warm-up on an NVIDIA RTX A4000 with 16 GB of memory.

| Choosers | NumPy CPU | GPU incl. transfers | GPU resident | End-to-end speedup | Resident speedup |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 5.156 ms | 0.845 ms | 0.361 ms | 6.10x | 14.26x |
| 100,000 | 50.614 ms | 3.587 ms | 1.037 ms | 14.11x | 48.79x |
| 1,000,000 | 499.375 ms | 36.610 ms | 8.118 ms | 13.64x | 61.51x |

For all three sizes:

- choice mismatches: **0**;
- maximum logsum error: less than **0.000001**; and
- the fused kernel avoided allocating the global utility matrix.

At one million choosers, this original comparison was about 13.64 times faster. But NumPy was built as a readable correctness reference, not the strongest possible CPU competitor. Phase 1 therefore added a fused Numba implementation that spreads choosers over all 24 physical CPU cores.

### Phase 1: comparison with the strongest CPU baseline

For the original 32-alternative, 16-feature shape, the 24-core fused CPU was much stronger:

| Choosers | Best 24-core CPU | GPU incl. transfers | Transfer-inclusive result | GPU resident |
|---:|---:|---:|---:|---:|
| 10,000 | 0.390 ms | 1.260 ms | GPU is 0.31x as fast | 0.810 ms |
| 100,000 | 5.560 ms | 4.170 ms | GPU is 1.33x faster | 1.190 ms |
| 1,000,000 | 59.260 ms | 34.520 ms | GPU is 1.72x faster | 7.110 ms |

This result is more realistic and less dramatic. The GPU loses on the smallest job because transfer and launch costs dominate. Its advantage grows with the amount of work.

Phase 1 also tested a shape based on mandatory tour scheduling in prototype MTC. It has 190 time alternatives and 69 feature rows:

| Choosers | Best 24-core CPU | GPU incl. transfers | Speedup | GPU resident | Choice mismatches |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 10.760 ms | 2.560 ms | 4.19x | 0.970 ms | 0 |
| 100,000 | 109.660 ms | 20.380 ms | 5.38x | 6.730 ms | 2 |

The wider calculation gives each transferred row much more arithmetic, which suits the GPU. The two differences at 100,000 rows occurred when random draws were less than three ten-millionths from a probability boundary. NumPy chose one option while both fused Numba and CUDA chose the immediately following option. This tiny but important result means the project needs a clear 64-bit or mixed-precision rule before claiming exact ActivitySim equivalence.

The wider test also discovered and fixed a real GPU synchronization bug. The original 32 alternatives fit inside one group of 32 GPU threads, called a warp. With 190 alternatives, several warps shared temporary memory. One warp could reuse that memory before another had finished reading it. A new synchronization barrier fixes the race, and tests above 32 alternatives prevent it from returning.

### Phase 2: replaying real ActivitySim scheduling data

Phase 1 copied the *shape* of tour scheduling, but its people and feature values were generated by a random-number program. Phase 2 uses real model records from ActivitySim's public `prototype_mtc` example.

ActivitySim's **mandatory tour scheduling** step chooses when work and school tours begin and end. For example, one option might leave at 8 a.m. and return at 5 p.m. The utility equation considers 52 to 59 active terms, including:

- the traveler's type, income, and work or school situation;
- the proposed start time, end time, and duration;
- whether another tour already occupies part of the day;
- a mode-choice logsum describing the quality of available travel modes; and
- constants that make some departure, arrival, and duration ranges more or less attractive.

The capture program observes ActivitySim at a documented internal boundary. It records the evaluated expression terms, feasible alternatives, coefficients, probabilities, random draws, and final selected positions. It does not change ActivitySim's rules or generate new draws.

The public sample produced six mandatory scheduling batches: first and second tours for work, school, and university students. Together they contain 4,477 tour choices. The largest batch contains 3,381 first work tours and 642,390 feasible chooser-alternative rows.

### Did the answers stay the same?

Yes. All 4,477 GPU choices matched ActivitySim exactly. A separate 64-bit replay of ActivitySim's probability tables also had zero mismatches. The largest batch's maximum utility difference was less than 0.00000044. In the full model output, every tour's selected time alternative, start time, and end time matched the earlier warm Sharrow run.

### Was the GPU faster?

The answer depends on where the input data begins:

| Copies of the real batch | Choosers | CPU | GPU including transfers | GPU already resident | Resident speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,381 | 8.241 ms | 15.545 ms | 0.961 ms | 8.57x |
| 2 | 6,762 | 20.346 ms | 32.298 ms | 1.326 ms | 15.34x |
| 4 | 13,524 | 43.035 ms | 62.427 ms | 2.242 ms | 19.20x |

For the larger rows, Phase 2 repeats captured real tours and their original draws. It does not invent synthetic travelers. Repeating records is a throughput test, not a claim that the public model contains more households.

The resident kernel clearly wins. The transfer-inclusive path clearly loses. The reason is that the current lowerer expands the expressions into a large table of term values: 151.6 megabytes for the native largest batch. Sending that table across the computer's PCIe connection takes much longer than the GPU arithmetic.

An analogy is a very fast bakery on the other side of town. Baking takes one minute, but delivering every separate ingredient takes fifteen minutes. Buying a faster oven will not solve the delivery problem. The next design must send compact ingredients once, prepare more of them inside the GPU, and keep them there for connected model steps.

### What has Phase 2 proved?

It has proved that the ragged scheduling kernel can process the real ActivitySim alternative sets with exact choices, and that its resident computation is faster than a strong CPU boundary on this machine. It has also disproved the idea that simply adding a GPU call around an expanded term table makes the isolated component faster.

At that Phase 3 boundary, it had not yet proved that the entire mandatory scheduling component or full ActivitySim model ran faster. Phases 4 through 6 later added those wider proofs. Stateful timetable expressions and other ActivitySim front-end work still run on the CPU.

### Phase 3: sending compact ingredients instead of finished terms

Phase 2 sent one value for every expression and every feasible alternative. Many values repeated the same traveler fact or the same time-alternative fact thousands of times. It was like shipping 59 finished ingredient bowls for every possible schedule.

Phase 3 sends a compact package instead:

- traveler facts are stored once per tour chooser;
- start, end, and duration are stored once for each of 190 time alternatives;
- each feasible row carries a small alternative ID;
- only the mode-choice logsum and seven timetable-dependent values stay row-specific; and
- ActivitySim still supplies the exact random draw.

The compact package for the largest real batch is 22.046 megabytes instead of 151.604 megabytes, an 85.46 percent reduction.

### What does the compiler do?

A **compiler** translates instructions from one form into another. Here it reads the real ActivitySim expressions, checks that every operation and column is supported, and writes matching CUDA arithmetic into the kernel. It supports the scheduling model's numbers, arithmetic, comparisons, and Boolean combinations such as "A and B." Unknown names and unsupported function calls cause a clear error instead of a silent wrong answer.

The same expressions also generate a strong CPU version compiled by Numba and spread across 48 CPU threads. This makes the comparison much harder and fairer than comparing the GPU only with a simple Python loop.

| Copies of the real batch | Choosers | Compact input | 48-thread CPU | GPU including transfers | GPU resident | Inclusive speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,381 | 22.046 MB | 9.481 ms | 2.478 ms | 0.181 ms | 3.83x |
| 2 | 6,762 | 44.091 MB | 18.443 ms | 4.681 ms | 0.271 ms | 3.94x |
| 4 | 13,524 | 88.179 MB | 37.132 ms | 9.348 ms | 0.410 ms | 3.97x |
| 8 | 27,048 | 176.355 MB | 69.735 ms | 17.787 ms | 0.709 ms | 3.92x |

All six mandatory-scheduling batches still have zero choice mismatches. The largest utility difference from the Phase 2 calculation is 0.0000229, and the largest GPU-versus-CPU logsum difference is 0.00000293.

Phase 3 proves transfer-inclusive superiority at this captured kernel boundary. It still does not time the work needed to build the compact tables inside ActivitySim, calculate mode-choice logsums, update the timetable, or map results back into pandas tables. Those steps belong in the next complete-component test.

### Phase 4: putting the GPU inside the real ActivitySim component

Phase 4 performs that complete-component test. ActivitySim now has an explicit setting that chooses either its normal scheduling calculation or ChoiceForge. It is not a hidden test hook. ActivitySim still decides which alternatives are possible, calculates mode-choice logsums, supplies the controlled random numbers, updates each person's timetable, and writes the output tables.

The first integrated version was correct but slower. A profile showed that the GPU work was not the problem. Repeatedly asking seven timetable questions for 642,390 rows took about 4.03 seconds. A day in this model has only 21 time slots, so ChoiceForge now answers those questions once on a small traveler-by-time-slot grid and looks up the required answers. That reduced the largest timetable stage to about 0.331 seconds.

Two three-trial experiments measured the result:

| Test | Cached Sharrow median | ChoiceForge median | Speedup | Schedule mismatches |
|---|---:|---:|---:|---:|
| Normal full-model runs | 5.515 s | 4.925 s | 1.120x | 0 |
| Same saved checkpoint before scheduling | 8.599 s | 8.014 s | 1.073x | 0 |

The timer includes the work that Phase 3 left outside: mode-choice logsums, timetable calculations, compact packing, ActivitySim's random-number call, transfers, GPU work, result mapping, and timetable updates. It even includes fresh-process GPU setup. All three normal ChoiceForge runs produced exactly the same 9,806 final `tdd`, start, end, duration, destination-logsum, and mode-choice-logsum values as the cached-Sharrow reference.

Why is the complete-component speedup 1.120x instead of 3.83x? The 3.83x number describes the calculation ChoiceForge directly replaces. The complete component also performs substantial mode-choice logsum and table work that remains the same in both versions. Speeding up one part of a larger job gives a smaller improvement to the whole job. This is an example of **Amdahl's law**: total speedup is limited by the work that was not accelerated.

### Phase 5: one backend for the scheduling family

Mandatory tours are only one kind of tour. A **non-mandatory tour** might be shopping, eating out, escorting someone, or recreation. A **joint tour** is shared by members of a household. An **at-work subtour** starts and ends at a workplace, such as going out for lunch. These components use related choice math, but they do not have identical columns or timetable rules.

Phase 5 taught the compiler to understand those differences without creating three separate GPU systems. It can turn text categories such as "escort" into compact yes-or-no columns, recognize both "equals" and "does not equal," and discover whether a timetable row belongs to a person or a tour. Calculations that are safe and repetitive go into the fused GPU kernel. Special state-changing timetable work remains explicit and checked.

Three complete-model trials measured all four ActivitySim scheduling timers:

| Scheduling component | Cached Sharrow median | ChoiceForge median | Speedup |
|---|---:|---:|---:|
| Mandatory | 5.515 s | 5.077 s | 1.086x |
| Joint | 0.671 s | 0.504 s | 1.331x |
| Non-mandatory | 1.560 s | 0.550 s | 2.836x |
| At-work subtour | 0.309 s | 0.202 s | 1.530x |
| **All four together** | **8.045 s** | **6.325 s** | **1.272x** |

The four-component scheduling family uses 21.4 percent less time, saving a median 1.720 seconds. This was not caused by one lucky run: the slowest ChoiceForge trial was still faster than the fastest cached-Sharrow trial. Non-mandatory scheduling improves most because it contains a large amount of repeated choice work but does not carry the same expensive mode-choice-logsum work as mandatory scheduling.

Correctness is checked more strongly than before. For every Phase 5 trial, the final accessibility, households, joint-tour participants, land use, persons, tours, and trips files have exactly the same SHA-256 fingerprints as the cached-Sharrow reference. A SHA-256 fingerprint is a long number made from every byte in a file; changing even one character almost certainly changes the fingerprint. This covers 9,806 tour rows and 23,583 trip rows, not just a few selected columns.

There was an important Phase 5 limit. Its separately run sessions did not prove an all-model gain because unrelated run-to-run variation was larger than the scheduling savings. Phase 6 later closed that evidence gap with an alternating A/B experiment.

## 8. A negative result that teaches an important lesson

The project also tested only ActivitySim's final probability-sampling operation for 100,000 choosers and 32 alternatives.

| Method | Time | Result compared with warm ActivitySim Numba |
|---|---:|---:|
| ActivitySim Numba on CPU | 2.515 ms | baseline |
| GPU including transfers | 2.591 ms | 0.97x, slightly slower |
| GPU resident | 0.212 ms | 11.86x faster |

There were zero choice mismatches. However, after paying to transfer the probability table, the isolated GPU call was slightly slower than ActivitySim's warm CPU function.

This is useful evidence, not a failure to hide. It shows that sending only the final tiny step to the GPU is a poor design. The strong performance idea depends on fusing utility evaluation, logsum calculation, and sampling, then keeping data on the GPU across connected steps.

## 9. How ActivitySim fits in

**ActivitySim** is a popular open-source activity-based travel modeling framework written in Python. ChoiceForge is designed as a possible specialized computing backend for parts of ActivitySim, not as a new travel demand model that replaces its behavioral rules.

The project integrated ChoiceForge with ActivitySim 1.4 and ran the public 25-zone prototype MTC example from start to finish:

- 5,000 households;
- 8,212 persons;
- 34 completed model steps;
- 70.568 seconds total runtime; and
- 580.8 MB peak unique memory.

That early 70.568-second run was only a framework baseline. Phase 4 later ran the full model with ChoiceForge handling mandatory scheduling, and Phase 5 expanded it to all four tour-scheduling components.

ActivitySim can also use **Sharrow**, an optimized expression system. Its compile run eventually completed in 27.3 minutes and used about 15 GB of unique memory. That compilation is not a fair production timing. After the cache existed, three warm runs took 61.650, 59.119, and 61.790 seconds, for a median of 61.650 seconds. Their final household, person, tour, and trip files were identical.

The warm profile showed where time is spent. Trip destination used 25.3% of total time. Mandatory tour scheduling used 9.1%, and trip scheduling used 8.4%. Mandatory scheduling is a practical first integration target, but it cannot reduce total runtime by 10% on its own because it is only 9.1% of the model. A reusable scheduling backend should target several scheduling components together.

### Phase 6: making destination work large enough

Trip destination asks where an intermediate stop happens. For every candidate
stop, the model also estimates how attractive the travel from the trip origin
to the stop would be and how attractive the remaining travel to the main tour
destination would be. Those are the two **directions** called OD and DP.

The first GPU replay exposed 30 small groups, separated by trip number and
purpose. Sending each group to the GPU separately was a bad bargain: the GPU
took about 15.032 milliseconds including transfers, while the CPU needed only
2.646 milliseconds. Think of sending 30 nearly empty delivery trucks instead
of loading one truck.

ChoiceForge then packed all 30 groups into one segmented kernel launch. A
segment label tells the kernel which purpose's coefficients to use, while
offsets tell it where each traveler's uneven list of candidate places begins
and ends. The packed job contained 3,971 choosers and 53,927 candidate rows.
It took 0.847 milliseconds on the GPU including transfers versus 1.241
milliseconds for a strengthened, batched Numba CPU version: **1.464 times
faster**. All 3,971 chosen positions matched, and logsums differed by at most
about nine millionths.

The crossover test explains when to use it. At 28,570 rows the GPU was slower.
At 39,706 rows it was faster. On this computer, this shape should be packed to
roughly 35,000 rows before GPU dispatch. That threshold must be remeasured on
other computers.

### The first proven whole-model improvement

The live ActivitySim loop still asks for one small purpose group at a time, so
turning on the new GPU sampler there would make it slower. Phase 6 instead
removed duplicated work from the much larger directional-logsum calculation.
It preserved ActivitySim's OD random draws, then its DP draws, but processed
the shared deterministic work together.

Six fresh full-model runs alternated the old and new paths in the order
A1/B1/A2/B2/A3/B3. Trip destination fell from a 16.307-second median to 14.238
seconds, **1.145 times faster**. All 34 model steps fell from 59.209 to 56.264
seconds, **1.052 times faster**. Every new-path component run beat every
old-path component run. Seven final result files were byte-for-byte identical
in every trial.

This distinction matters: the 1.464x number proves the packed CUDA kernel
boundary; the 1.145x number proves the complete destination component; and the
1.052x number proves this complete example model. They are not interchangeable.

### Phase 7: stop repeating setup, then accelerate the actual nest

Phase 7 asked what was still consuming the 14.238 seconds. ActivitySim repeated
a 70-step preparation recipe once for each of ten trip purposes, even though
most of that recipe was the same. ChoiceForge now puts the purposes for one
trip number into one larger worksheet. The recipe runs three times - once for
trip number 1, 2, and 3 - instead of 30 times. Sampling and final decisions remain
separate, so each traveler keeps the same random draws and behavioral rules.

The next boundary is called **nested logit**. Imagine grouping 21 travel modes
into folders: driving modes, walking/biking, transit, and ride-hailing. Some
folders contain smaller folders. Nested logit combines the scores inside each
folder and then combines the folders. It represents that two similar choices,
such as two transit services, are more closely related than transit and driving.

We captured every real 21-mode score table used by destination choice: 30
batches and 107,854 rows. ActivitySim's dataframe reducer needed a median of
486.626 milliseconds for the complete captured sequence. The fused GPU kernel,
including sending 18.119 MB to the GPU and returning the answers, needed 120.737
milliseconds in the longer 31-trial test: **4.030 times faster**. Even its slowest recorded GPU trial was
faster than the fastest CPU trial. Its largest logsum difference was about
0.0000000000000036.

The full-model test alternated the Phase 6 and Phase 7 programs six times.
Trip destination fell from 12.179 to 10.281 seconds, **1.185 times faster**.
All 34 model steps fell from 54.459 to 52.166 seconds, **1.044 times faster**.
Every Phase 7 run beat every Phase 6 run, and all seven final result files were
byte-for-byte identical in all six trials.

Why is a 40x kernel only a 1.044x whole-model improvement? The kernel replaces
less than half a second of CPU reduction. Most remaining time evaluates 404
behavioral expressions, samples destinations, runs other model components, and
organizes data. This is Amdahl's law: speeding up a small slice cannot speed up
the whole pie by the same amount.

### Phase 8: current software and ten times more households

Earlier complete-model tests used 5,000 households. Phase 8 asked two harder
questions. Does the integration still work with a pinned current ActivitySim
development version? Does the advantage remain when the public workflow grows
to 50,000 households across 190 zones?

The answer on this workstation is yes. The experiment alternated three regular
ActivitySim runs with three ChoiceForge runs, always starting a fresh process:

| Measured boundary | Regular ActivitySim median | ChoiceForge median | Speedup |
|---|---:|---:|---:|
| Four scheduling components | 30.1 s | 23.7 s | **1.270x** |
| Trip destination | 30.9 s | 21.8 s | **1.417x** |
| All 34 model steps | 147.225 s | 131.812 s | **1.117x** |

ChoiceForge saved a median of 15.413 seconds for the whole run. Every one of
the three ChoiceForge runs was faster than every regular run. The seven final
files matched byte for byte in all six trials. Those files describe 50,000
households, 111,130 people, 142,761 tours, and 350,751 trips.

The scale also shows what happens inside the computer. The largest scheduling
call contains 9,561,750 feasible schedule rows for 50,325 choosers. Its compact
input is 327.920 megabytes. The GPU part takes about 0.4 seconds, but preparing
timetable facts, packing the data, retrieving controlled random numbers, and
mapping the answer make the entire boundary about 3.3 seconds. This means the
next scheduling improvement should focus on preparation and data reuse, not
only on making the GPU arithmetic faster.

Phase 8 also captured a much larger nested-logit calculation: 30 real batches,
3,186,130 rows, and 535.270 megabytes of 64-bit mode scores. In 11 alternating
trials, ActivitySim's CPU reducer needed a 4.646627-second median. The GPU,
including both transfers, needed 0.125627 seconds: **36.988 times faster**.
Every GPU trial beat every CPU trial, and the largest logsum difference was
about 0.0000000000000053.

That 36.988x result is for one captured mathematical boundary. The 1.117x
number is the honest complete-model result. The rest of ActivitySim still has
to create people and tours, evaluate expressions, sample places, organize
tables, and write outputs.

### Phase 9: much larger geography and a useful stop sign

Phase 9 downloaded the public full-geography Prototype MTC data. The full data
has 2,875,192 households, 7,566,527 people, and 1,454 zones. That is 7.7 times
as many zones as Phase 8. A zone is a geographic area such as a neighborhood
or traffic-analysis area, so more zones make the travel calculations much
larger and more varied.

The whole population needs far more memory than this workstation has. The
published configuration describes a 64-process, 432-gigabyte-RAM setup. The
local test therefore used 50,000 households but kept all 1,454 zones. It ran
one fresh regular ActivitySim process and one fresh ChoiceForge process.

For this test, ChoiceForge accelerated only trip destination. It left the
schedule choices on the regular ActivitySim path. The final accessibility,
household, participant, land-use, person, tour, and trip files were byte for
byte identical. They describe 50,000 households, 132,536 people, 175,579 tours,
and 442,682 trips.

| Measured boundary | Regular ActivitySim | ChoiceForge | Observed result |
|---|---:|---:|---:|
| Trip destination | 39.2 s | 28.0 s | **1.400x faster** |
| All 34 model steps | 198.794 s | 187.010 s | **1.063x faster** |

This is one correctly matched pair, not yet a final statistical claim. A
high-memory computer must run several alternating pairs before those numbers
can be called reliable medians.

Phase 9 also found nine schedule choices that changed at a floating-point
boundary when the experimental scheduling GPU path was enabled at 100,000
households. One later decision changed because the earlier difference affected
the model's sequence of controlled random numbers. ChoiceForge therefore turns
that scheduling path off for this configuration. This is not a failure to hide:
it is exactly why the project checks complete output files rather than trusting
a fast-looking timer. A faster answer that changes the model is not acceptable.

## 10. What the current evidence proves

The project can currently support these narrow statements:

- The fused linear-choice GPU kernel runs correctly on the tested NVIDIA hardware and software setup.
- It produced the same choices as the CPU reference in the reported synthetic benchmarks.
- Its logsums were numerically extremely close to the reference.
- For the reported large synthetic workload, it was much faster than the readable, materializing NumPy reference, even when transfers were included.
- For the scheduling-shaped 10,000-row workload, it was 4.19 times faster than a fused 24-core Numba CPU including transfers, with zero choice mismatches.
- The GPU advantage depends strongly on problem shape and size: it lost to the fused CPU on the 10,000-row, 32-alternative workload.
- Avoiding a full utility matrix can reduce intermediate memory use.
- ActivitySim-compatible random draws and final sampling rules can be preserved.
- Offloading sampling alone is not worthwhile when transfer time is included.
- Warm Sharrow production runtime and component shares are now measured reproducibly.
- Phase 2 replayed 4,477 real mandatory-tour choices with zero GPU mismatches.
- The real scheduling kernel is 8.57 times faster when inputs are GPU-resident, while the transfer-inclusive path is slower at 0.53 times CPU speed.
- Phase 3 reduces that input by 85.46 percent and makes the native transfer-inclusive GPU boundary 3.83 times faster than a generated 48-thread CPU implementation.
- Phase 4 makes the complete mandatory scheduling component 1.120 times faster than cached Sharrow in normal runs and 1.073 times faster in matched-checkpoint runs.
- Phase 5 makes the complete four-component scheduling family 1.272 times faster, led by a 2.836 times non-mandatory scheduling speedup.
- Seven substantive final CSV files are byte-identical across all Phase 5 trials, including all 9,806 tours and 23,583 trips.
- Phase 6 makes the packed real destination kernel 1.464 times faster than a batched Numba CPU including transfers, with zero mismatches across 3,971 choices.
- Phase 6 makes the complete trip-destination component 1.145 times faster and the complete 34-step prototype model 1.052 times faster in interleaved trials.
- Seven substantive final CSV files are byte-identical across all six Phase 6 A/B trials.
- Phase 7's transfer-inclusive nested-logit reducer is 4.030 times faster than ActivitySim's pandas reducer at the captured aggregate boundary, with maximum absolute error of 3.6e-15.
- Phase 7 makes trip destination 1.185 times faster and the complete example model 1.044 times faster than Phase 6; every optimized run beats every baseline run.
- Phase 16 makes the generated-utility trip-destination component 1.025 times faster in three repeated 50,000-household public pairs; every candidate destination time beats every baseline time and every modeled decision matches.
- Phase 17 strengthens that result to a 1.040-times destination speedup in five repeated pairs, with a positive 0.8-to-1.9-second bootstrap interval and zero changed modeled decisions.
- Phase 17's complete-model median is 1.006 times faster, but its bootstrap interval still includes a 0.042-second slowdown, so the strict whole-model gate remains failed.
- Seven substantive final CSV files are byte-identical across all six Phase 7 A/B trials.
- Phase 8's 3,186,130-row nested-logit replay is 36.988 times faster on the GPU including transfers, with maximum absolute error of 5.4e-15.
- On pinned current ActivitySim at 50,000 households, the four scheduling components are 1.270 times faster, trip destination is 1.417 times faster, and all 34 models are 1.117 times faster.
- All three Phase 8 optimized runs beat all three baselines, and seven substantive CSVs are byte-identical across all six runs, including all 142,761 tours and 350,751 trips.
- On public full MTC geography with 1,454 zones, Phase 9's destination-only GPU configuration has byte-identical outputs in one 50,000-household A/B pair and an observed 1.400x trip-destination ratio. It is a scale gate, not a median claim.
- The Phase 9 scheduling experiment found nine changed choices at 100,000 households, so that GPU path is disabled for full geography until a precision safeguard resolves them.

## 11. What the current evidence does not prove

It does **not** yet prove that:

- every ActivitySim model will run faster;
- every complete model will be faster;
- whole-model superiority for the Phase 17 generated-utility backend; its median is faster, but two nearly tied pairs and an interval including zero keep the strict gate from passing;
- a complete model will be 13.64 times faster;
- arbitrary ActivitySim utility expressions can run in the kernel;
- destination choice with thousands of irregular alternatives is solved;
- results will be identical on every GPU and software version; or
- a faster calculation makes the behavioral model more accurate.

The NumPy reference remains useful for correctness, and Phase 1 includes a strong fused 24-core CPU competitor. Phase 2 supplies the real scheduling replay, Phase 3 supplies transfer-inclusive scheduling-kernel superiority, Phase 4 supplies the first complete-component comparison, Phase 5 supplies breadth, Phase 6 supplies a destination replay, Phase 7 supplies batched setup plus a live nested-logit GPU boundary, Phase 8 supplies pinned-current 50,000-household evidence, and Phase 9 supplies a full-geography scale gate. A broader claim still requires a differently structured public model, other GPUs, high-memory repetition, and independent reproduction.

## 12. From promising prototype to convincing result

A responsible roadmap looks like this:

### Step 1: Expand the supported math - completed for four scheduling components

Phase 3 compiles the real scheduling arithmetic and Boolean expressions directly from compact traveler and alternative columns. Phase 5 adds categorical assignments, inequalities, and different timetable owners. Stateful timetable functions remain precomputed primitives, and other ActivitySim components may require more operations.

### Step 2: Build a configured ActivitySim backend - completed for the tour-scheduling family

Phase 4 adds an explicit setting, preserves ActivitySim-owned random draws and timetable updates, and falls back to ActivitySim for unsupported tracing or estimation paths. Phase 5 reuses that contract for four scheduling components.

### Step 3: Handle large destination choices - completed for the prototype model

Phase 6 packs 30 real ragged destination segments into one launch and proves a transfer-inclusive GPU win. Phase 7 batches live preprocessing across purposes and adds the 21-mode nested-logit GPU reducer. Much larger regional alternative sets remain future work.

### Step 4: Prove correctness on public benchmark data - completed for the prototype scheduling family

Phase 2 checks all six prototype MTC mandatory-scheduling batches and records every numerical tolerance. Phase 5 verifies byte-identical final files after four scheduling components. Phase 6 checks all 30 destination batches and hashes all six A/B model trials. Phase 8 repeats the proof at 50,000 households on current ActivitySim. Phase 9 repeats the destination test across 1,454 public MTC zones and disables the scheduling path when its exactness gate fails. A differently structured public model is still needed before making a general claim.

### Step 5: Run fair performance comparisons - completed for four scheduling components

Phase 5 compares three cached-Sharrow and three transfer-inclusive ChoiceForge scheduling runs. Phase 6 strengthens the design with alternating A/B processes and proves both component and whole-model gains. Other models and hardware still need the same treatment.

### Step 6: Publish a reproducible evidence package

Include exact commands, pinned dependencies, raw timing samples, correctness reports, benchmark data instructions, and results from more than one GPU. Let other researchers reproduce or challenge the claim.

## 13. Why faster travel modeling could matter

Faster computation does not automatically make a better transportation decision. But it can change what analysts have time to study.

### More scenarios

Instead of testing only a few projects, an agency could examine more combinations of fares, services, land use, road pricing, and policy assumptions.

### Better uncertainty analysis

No forecast is certain. Faster models can support many runs with different population growth, fuel prices, remote-work rates, or random seeds. The range of results may be more informative than one forecast.

### Faster feedback

Modelers could find data or configuration problems sooner, shorten development cycles, and respond more quickly to community questions.

### Lower intermediate memory pressure

Fused calculations that avoid giant temporary tables may allow larger problems or reduce memory bottlenecks.

### Important cautions

Speed cannot repair biased survey data, missing travel options, unrealistic assumptions, or unfair project goals. GPUs use electricity and require specialized hardware and skills. Public decisions still need transparency, equity analysis, local knowledge, and human judgment.

## 14. Frequently asked questions

### Does the GPU predict travel differently?

It should implement the same model rules. The goal is to calculate them faster, not invent a different behavior model.

### Does zero choice mismatch mean every number is identical?

Not always. Some kernel-level logsums differ by tiny amounts because floating-point arithmetic can be ordered differently in parallel, even when the selected alternative is unchanged. In Phases 5, 6, and 8, however, all seven substantive final CSV files were byte-for-byte identical in the reported A/B trials.

### Why use 32-bit numbers?

They use half the memory of 64-bit numbers and are usually faster on GPUs. Whether they are accurate enough must be tested for each model. ActivitySim-compatible final sampling also has a 64-bit path in the adapter.

### Why not move the entire model to the GPU immediately?

ActivitySim contains varied expressions, tables, joins, tracing, estimation tools, and irregular choice sets. A small, well-tested component creates a safer foundation than a large rewrite whose errors are hard to locate.

### Is a GPU always faster?

No. Small workloads, frequent transfers, branching logic, or underused cores can make a GPU slower. The isolated sampling benchmark demonstrated this directly.

### Which speedup is the headline?

Use the number for the boundary being discussed. Phase 3's scheduling kernel is 3.83x, Phase 7's smaller nested-logit reducer is 4.030x, and Phase 8's much larger nested-logit replay is 36.988x. At the 50,000-household scale, Phase 8 makes the four complete scheduling components 1.270x faster, complete trip destination 1.417x faster, and the complete model 1.117x faster. Phase 9's 1.400x destination ratio uses much larger 1,454-zone geography, but is only one matched pair. Only a carefully repeated full-model number should be called a whole-model speedup.

## 15. Mini glossary

| Term | Plain-English meaning |
|---|---|
| ActivitySim | An open-source framework for activity-based travel demand models. |
| Amdahl's law | The idea that total speedup is limited by parts of a job that were not accelerated. |
| Alternative | One possible choice, such as walk, bus, or car. |
| Availability | A rule saying whether an alternative is possible. |
| Benchmark | A controlled experiment that measures speed and correctness. |
| Chooser | The person, household, tour, or trip making a simulated decision. |
| Compiler | A program that translates checked instructions into another executable form. |
| Compact representation | Data stored without unnecessarily repeating the same facts. |
| CPU | A flexible general-purpose processor with a modest number of powerful cores. |
| CUDA | NVIDIA's platform for programming its GPUs. |
| Feature | An input fact, such as travel time, cost, income, or vehicle access. |
| Floating point | The computer's approximate way of storing non-whole numbers. |
| Fusion | Combining connected calculations so intermediate data is not repeatedly stored or moved. |
| GPU | A processor with many simpler cores suited to large parallel calculations. |
| GPU kernel | A small program executed by many GPU threads. |
| Logsum | A summary of the attractiveness of all available alternatives. |
| Nested logit | A choice calculation that groups related alternatives before combining them. |
| Probability | A number from 0 to 1 describing how likely an outcome is. |
| Shared memory | Small, fast GPU memory close to threads in one block. |
| Skim | A table of travel time, distance, cost, or related values between places. |
| Synthetic population | Anonymous simulated households and people resembling a real region. |
| Utility | A model score representing how attractive an alternative is. |

## 16. The bottom line

ChoiceForge asks a focused question: can a common bundle of travel-choice calculations be redesigned as one correctness-first GPU kernel that avoids huge intermediate tables?

The answer is now more precise. The fused kernel can decisively beat a strong 24-core CPU when each chooser has enough alternatives and features, but it loses on small, narrow jobs. Phase 1 also found a multi-warp synchronization defect and two near-boundary choice differences, demonstrating why wider correctness tests and precision rules matter as much as speed.

Phases 3 through 9 advanced integration through full MTC geography. Phase 8 improved whole-model time with exact outputs. Phase 9 confirmed exact destination results across a far larger zone system and held scheduling back until precision improves. Phase 11 then repeated that full-geography destination test and made the outcome stronger. Phase 12 identified the utility-calculation opportunity. Phase 13 completed the exact CPU answer key. Phase 14 generated the matching GPU evaluator and proved exact agreement on real public-model batches, while keeping it safely in shadow mode.

## 17. The latest evidence: replicated destination speedup and a safer path to GPU utilities

This section updates the earlier phases. It uses two labels deliberately:

- **Proven production result** means a complete public ActivitySim run finished, its final files matched exactly, and the timing was repeated.
- **Foundation or microbenchmark** means a smaller, controlled engineering test succeeded. It is useful evidence, but it is not yet a claim that a whole ActivitySim model got faster.

### Phase 10: a guardrail, not a shortcut

The full-geography scheduling experiment revealed a subtle risk. A GPU and CPU can both follow the same-looking formula yet round a last digit differently. If a random draw lands exactly near the boundary between two choices, that last digit can change the selected alternative. One changed early choice can also change the controlled sequence of later random draws.

Phase 10 added a **shadow guard**. Think of it as a second answer key running beside an experimental route. The established CPU/Sharrow result remains authoritative. The GPU is allowed to report success only if its shadow comparison passes; otherwise the system deliberately returns the CPU result. This is slower than blindly using the GPU, but it prevents a silent model change while the arithmetic issue is investigated.

### Phase 11: the current headline result

Phase 11 focused on the part that was already exact: trip destination. It used the public Prototype MTC full geography with 1,454 zones and 50,000 households. Three fresh, interleaved A/B pairs were run: regular ActivitySim, then ChoiceForge, with the order alternated to reduce the chance that unrelated computer activity picked the winner.

| Measured boundary | Regular ActivitySim median | ChoiceForge median | Result |
|---|---:|---:|---:|
| Trip destination | 39.7 s | 28.6 s | **1.388x faster** |
| All 34 model steps | 202.492 s | 190.380 s | **1.064x faster** |

The complete run saved a median of 12.112 seconds. The three paired whole-model savings were 12.172, 12.112, and 7.596 seconds. A simple resampling check placed the median saving between 7.596 and 12.172 seconds for these three pairs. Every optimized run was faster than its paired baseline.

Most importantly, all six substantive final CSV output files were **byte-for-byte identical** in every pair. That is stronger than saying the choices merely looked similar: the saved files matched character for character. The experiment also pins the exact ActivitySim source revision, configurations, patches, and hashes, so another person can repeat the same setup.

This is the honest current headline: on this workstation, for this public 50,000-household full-geography workload, ChoiceForge made the complete model 1.064 times faster while reproducing its reported final outputs exactly. It does not mean every ActivitySim model, computer, or future version will gain the same amount.

### Why the next bottleneck is utility calculation

Destination choice works in two broad stages. First, the model evaluates many utility equations: for each potential destination it combines travel time, cost, household facts, destination facts, and rules. Second, it reduces those scores into probabilities and a selected destination.

ChoiceForge already makes the second stage faster. But the Phase 11 profiler showed that preparing and sending the first stage's finished utility values involved about 12,564,936 rows, or roughly 2.111 gigabytes of data. Sending a huge finished answer sheet to the GPU is like carrying every completed math problem to a very fast calculator. The more ambitious improvement is to send the compact ingredients and calculate the utility scores on the GPU too.

### Phase 12: proving the ingredients can be translated safely

ActivitySim utility equations are written as expressions such as "travel time times a coefficient, plus an income term, only when a condition is true." A GPU cannot safely accelerate these expressions merely by guessing what they mean. The order of arithmetic, special functions, missing values, and table lookups all matter.

Phase 12 therefore built three safety layers:

1. An **expression reader** turns the public trip-mode-choice formulas into a small checked tree of operations. It successfully parses and evaluates all 253 distinct expressions used in that configuration on both CPU and GPU test paths.
2. A **skim adapter** gives the GPU path access to travel-time and cost lookup tables in the same way the model asks for them. Its tests confirm the adapter returns the expected values.
3. A **shadow capture** runs the GPU calculation beside the trusted CPU/Sharrow calculation without changing the official model answer. It records any difference for investigation.

The good news is that the deterministic GPU kernel exactly matched the local ordered CPU expression evaluator in the shadow test. The caution is that a compiled path can use different input precision, operation grouping, reduction order, or optional speed transformations such as `fastmath`. Any of these can change rounding in the last few binary digits. The shadows found differences from tiny fractions up to 0.25 for extremely large "unavailable" scores. No GPU utility result is allowed to replace Sharrow's production result until one explicit cross-device policy controls both targets.

### A controlled speed test: encouraging, but not a production claim

Phase 12 also tested a deliberately simple 250,000-row, 64-feature, 21-alternative utility-and-choice pipeline. The CPU NumPy reference took 235.215 milliseconds. The GPU lowering plus 21-mode reduction took 28.988 milliseconds, an **8.114x** pipeline speedup, while the logsum check stayed within 0.00000000001.

This result answers a narrow question: can the compact-ingredient design be fast enough to matter? Yes. It does **not** yet prove an 8.114x whole-model improvement, because the test is not the full ActivitySim destination component and the CPU reference is not Sharrow's compiled production evaluator.

### The strict IR plan: one recipe, two kitchens

The project is now pursuing a more rigorous solution rather than trying to tune away discrepancies one at a time. A **strict intermediate representation**, or strict IR, is a precise shared recipe for an equation. Instead of separately interpreting an expression for the CPU and generating a similar-looking CUDA kernel for the GPU, both targets are built from the same checked list of operations and numeric rules.

An everyday analogy: two kitchens can make the same named dish but use different measuring cups and a different order of steps. A strict recipe states the ingredients, measurements, order, and rounding rules so both kitchens produce the same dish. For ChoiceForge, the two kitchens are the strict CPU evaluator and the generated CUDA target.

The strict IR has generated a canonical description of the public MTC trip-mode utility: 379 terms across 21 alternatives. Phase 13 completed the strict CPU target, including separate ordered multiply and add steps, exact comparison reports, and fail-closed policy checks. Phase 14 completed the CUDA target generated from the same IR. The project test suite now passes 95 tests, including exact cross-device edge cases, compact skim gathering, a device-resident handoff to the nested-logsum reducer, the Phase 16 FP32 arithmetic policy, Phase 17 plan reuse, and Phase 18 GPU-native runtime checks.

This remains a project implementation, not a completed upstream Sharrow feature. Phase 15 removed the qualification-only utility transfer but failed its 50,000-household scale gate. Phase 16 recovered a repeated large destination-component win. Phase 17 added persistent plans and trip-mode continuation, strengthening that component win and making the five-run whole-model median faster. The strict path remains opt-in because the whole-model interval still includes zero and replication on other hardware and models is unfinished.

### How to read the project status today

| Question | Best current answer |
|---|---|
| Is there a real, repeated, exact speedup? | Yes. Phase 11 achieved 1.064x for the complete 50k-household, 1,454-zone public run, with byte-identical outputs in three pairs. |
| Is the destination component itself faster? | Yes. Its Phase 11 median was 1.388x faster. |
| Has the large utility equation been connected to the real GPU path? | Yes. Phase 17 reuses generated plans for destination and trip-mode utilities, with Sharrow fallback and explicit telemetry. |
| Is the compact GPU utility approach promising? | Yes. It was 8.114x faster in a controlled microbenchmark with a tight numerical gate. |
| What prevents a bigger claim today? | The destination component wins, but whole-model timing noise still crosses zero; a second GPU and a second public model have not been tested. |
| What changed in Phase 16? | An explicit FP32 policy, compact inputs, and caching produce a repeated 1.025x large destination-component win. The whole-model gate still fails. |
| What changed in Phase 17? | Checked persistent plans and trip-mode continuation produce a five-pair 1.040x destination win and a 1.006x faster whole-model median; the strict whole-model gate still fails. |

## 18. Phase 13: building the exact CPU answer key

Phase 12 discovered that saying "use the same formula" was not precise enough. Two compilers may change when a number is rounded or combine a multiplication and addition. The answers usually remain close, but a travel model can place a random draw very near a choice boundary. That makes the last few digits important.

Phase 13 turns the arithmetic rules into a written and executable contract. The strict CPU evaluator is like an official answer key. It requires:

- 64-bit arithmetic while an expression is being calculated;
- one conversion of the finished feature to a 32-bit number;
- 32-bit coefficients and utility totals;
- the original term order;
- a separate multiplication and addition for every term; and
- normal IEEE rounding, with speed shortcuts disabled.

If the IR version, policy, or identifying hash changes unexpectedly, the evaluator refuses to run. It also refuses unresolved coefficients and malformed arrays. These are useful failures: they stop a plausible-looking but undefined calculation from entering the model.

### Did it cover the real model?

Yes. The canonical public MTC utility contains 379 terms for 21 travel-mode alternatives. Every term and alternative executes under the strict policy. An independent, simple scalar loop produced exactly the same utility bits. The complete repository now passes 95 tests.

Phase 13 then ran the public full-geography model with 1,001 households. It observed 30 real trip-mode batches containing 85,126 rows. That meant comparing 32,262,754 individual feature values and 1,787,646 utility values. ActivitySim completed all 34 model steps normally in 95.511 seconds, and Sharrow remained the official source of every model answer.

### What did the comparison teach us?

The new strict answer key and current Sharrow did not match exactly. They agreed on 99.9118 percent of feature cells and 66.8085 percent of utility cells. The largest feature difference was about 0.0000305. The largest utility difference was 0.25, occurring among very large scores where a float32 number has relatively wide spacing.

Those percentages do not mean Sharrow is "66.8 percent correct." Sharrow follows its existing compiled arithmetic, while the strict evaluator follows the newly published cross-device policy. Many different utility cells can result from one tiny feature difference or a different order of hundreds of additions. Phase 13's job is to expose and classify that fact, not silently declare one existing compiler's behavior universal.

Every observed difference now has a stage. The reports found 28,466 feature cells whose compiled-expression result did not follow the strict expression policy. Among 593,346 different utility cells, 183 were explained by those input differences alone, while 593,163 also reflected a different accumulation rule. The first feature difference was about 0.00000763; the first utility difference was one float32 step.

### Why Phase 13 is successful even though Sharrow differs

The goal was to create one stable answer key for future CPU and GPU code, not to copy whatever rounding happens on this workstation. All 379 terms and 21 alternatives run under that answer key. The comparison gate identifies the first different row, expression, alternative, values, and arithmetic stage. It can operate in observation mode, where Sharrow remains authoritative, or exact mode, where any difference stops qualification.

Phase 14 has now generated CUDA from this same strict IR and matched the strict CPU arrays exactly. Current Sharrow remains the safe ActivitySim fallback. The project will attempt another complete-model performance claim only after the generated path is integrated without qualification overhead and passes repeated byte-identical trials.

## 19. Phase 14: giving the GPU the exact same recipe

Phase 13 made the official answer key. Phase 14 built a code generator that
turns that same checked recipe into a CUDA program. This matters because two
separately handwritten programs can look equivalent while making tiny
different choices about types, rounding, or operation order. Generating both
evaluators from one hashed recipe greatly reduces that ambiguity.

### What does the generated GPU program do?

For each model row, one GPU work group evaluates all 379 travel terms. It stores
those temporary feature values in fast shared memory. Then separate GPU workers
calculate the 21 travel-mode scores. Each worker uses the coefficients in the
original order and performs a separate 32-bit multiplication and addition for
every term. The resulting 21 scores can remain on the GPU and go directly into
the nested-logsum calculation instead of taking a round trip through ordinary
computer memory.

Inputs are packed by their real meaning: decimal numbers, whole numbers, and
true/false values are not silently mixed. The compiled program is cached using
the recipe hash, input types, coefficients, generated source, and compiler
rules. Change any important ingredient and ChoiceForge builds a different
kernel instead of reusing a stale one.

### A tiny-number rule discovered by testing

Phase 14 found a useful edge case. A 32-bit floating-point number can be so tiny
that it enters a special range called **subnormal**. The NVIDIA runtime on this
machine turns such numbers into signed zero during these calculations. This is
called **flush to zero**.

Ignoring the behavior would leave the phrase "exactly the same arithmetic"
with a hidden exception. Instead, strict IR version 3 publishes the rule. The
CPU answer key now also turns float32 subnormals into signed zero after the same
steps. Tests deliberately use subnormal numbers, NaN ("not a number"), and
infinity. This makes the cross-device agreement a real contract rather than an
agreement that only holds for convenient inputs.

### Did it match on the public model?

Yes. The same public 1,001-household, full-geography ActivitySim workload
produced 30 trip-mode batches and 85,126 rows:

| Exact Phase 14 check | Result |
|---|---:|
| Batches | 30 of 30 |
| Feature values | 32,262,754 of 32,262,754 |
| Utility values | 1,787,646 of 1,787,646 |
| Largest feature difference | 0.0 |
| Largest utility difference | 0.0 |
| Terms and alternatives per batch | 379 and 21 |

Every checked GPU bit matched the strict CPU answer key. ActivitySim still used
Sharrow's established output as the official answer, so the experiment could
not alter model behavior. That is the safe way to qualify a new backend.

### Is it faster?

The generated kernel's diagnostic median was about 5.868 milliseconds. Preparing
and transferring its inputs took about 116.157 milliseconds, and downloading
the large comparison results took about 1.593 milliseconds. These numbers are
not a fair speed contest. Phase 14 intentionally captures every one of the 379
feature columns so it can prove exactness term by term, while production code
would keep compact inputs and useful outputs on the GPU.

Therefore the honest result is: **Phase 14 proves exact generated CPU/GPU
semantics, not a new end-to-end speedup.** A separate test already confirms
that the generated 21-column utility matrix can stay on the GPU through the
nested-logsum reducer with no utility download and no reducer re-upload.

### What does this change?

Before Phase 14, different arithmetic rules blocked the more ambitious GPU
utility path. That obstacle is now resolved for this IR and hardware stack.
That was the remaining Phase 15 problem: remove shadow-only transfers, connect
generated utilities to the real path, and repeat public-model A/B runs. The
next section explains what happened when those tests were completed.

## 20. Phase 15: putting the exact GPU recipe into the real model

Phase 14 proved that the CPU and GPU could follow the same recipe. Phase 15
asked a harder question: can the GPU's answer actually replace Sharrow's answer
inside ActivitySim, stay on the graphics card for the next calculation, and
make the real model faster?

### The first connection worked, but exposed a data problem

The first candidate made all 379 ingredients as ordinary columns before the
GPU kernel ran. Imagine copying every number from a library of road-time maps
onto one enormous worksheet, sending the worksheet to a calculator, and then
throwing it away. The arithmetic was fast, but preparing the worksheet was not.

At 1,001 households, the first repeated combined test looked faster than plain
ActivitySim. A fairer test then compared Phase 15 directly with Phase 11, so
both sides received the same destination batching and GPU nested-logsum work.
That test showed the first strict utility path itself was slower. Measuring the
parts found the cause: looking up and materializing skim columns took about
15.35 seconds across three candidate runs, while the useful GPU work took much
less time.

### The compact skim design

ChoiceForge was changed so the generated kernel receives:

- the shared road-time and cost map cubes;
- the already-checked origin, destination, and time positions for each row;
- the smaller ordinary chooser columns; and
- the strict coefficients and recipe.

The kernel now looks up a needed map value directly. It does not build a giant
379-column feature table. Identical map cubes are uploaded only once, even when
ActivitySim creates new temporary wrapper objects. After trip destination is
finished, the cache is released so later model steps do not pay for unused GPU
memory. If any part fails, ChoiceForge uses Sharrow instead.

### Did compact GPU gathering stay exact?

Yes. The final qualification covered 30 real public-model batches:

| Check | Exact result |
|---|---:|
| Rows | 85,126 |
| Feature values | 32,262,754 of 32,262,754 |
| Utility values | 1,787,646 of 1,787,646 |
| Largest strict CPU/GPU difference | 0.0 |
| Utility downloads before nested logsum | 0 bytes |
| Nested-logsum utility re-uploads | 0 bytes |

Every modeled trip decision also matched. A printed diagnostic called
`destination_logsum` differed by at most 0.000008 at this scale. That field is
not used as a later decision in this output check. The difference comes from
the published strict arithmetic versus current Sharrow arithmetic; the strict
CPU and GPU answers still match bit for bit.

### The direct repeated speed result

Three fresh pairs compared Phase 11 with compact Phase 15 at 1,001 households.
The destination component improved from a median 11.3 seconds to 10.3 seconds,
or **1.097 times faster**. Every Phase 15 destination run was faster than every
Phase 11 destination run. All 90 strict utility batches stayed on the GPU for
the nested calculation, and all modeled decisions matched.

The complete model did not pass the stricter promotion rule. Its median was
84.631 seconds for Phase 11 and 85.445 seconds for Phase 15. Other small model
steps varied enough to hide the one-second destination gain. ChoiceForge
therefore claims component superiority at this qualification scale, not a new
whole-model record.

### What happened at the large scale?

The final diagnostic used 50,000 households, all 1,454 zones, and 4,188,312
utility rows. Decisions still matched, but compact Phase 15 took 33.3 seconds
for trip destination versus 28.4 seconds for Phase 11. The complete model took
197.871 versus 192.519 seconds. This is a clear rejection, so running three
more expensive pairs would not create a responsible speed claim.

The reason is now narrower and useful. The GPU kernel gives one block to each
row and makes many scattered reads from large skim cubes. At 50,000 households,
that access pattern is slower than Sharrow's compiled expression path. The
next compiler should process tiles of nearby rows, combine repeated map reads,
reuse a gathered value across multiple terms, and keep the exact ordered
arithmetic rules.

### What should a high-school reader conclude?

An experiment can succeed even when the final answer is "do not deploy this
version." Phase 15 produced working, exact, fail-safe GPU integration and a
repeated component win. It also used a public large benchmark to prove where
the design stops winning. The production headline therefore remains Phase 11's
repeated 50,000-household result. Phase 15 is the tested bridge to a better
future compiler, not an excuse to replace a faster working system today.

## 21. How the complete calculation fits together

The earlier sections introduce each idea separately. This is the whole trip
destination calculation in one view:

```text
public households, people, zones, and travel-time maps
                         |
                         v
ActivitySim creates rows representing possible trip destinations
                         |
                         v
utility equations score 21 travel modes for each row
                         |
                         v
nested logsum summarizes how useful those modes are
                         |
                         v
destination probabilities are formed and one destination is selected
                         |
                         v
the selected trip and diagnostic values are written to final output files
```

Phase 11 accelerates the preparation and nested-logsum part of this pipeline.
Phase 15 additionally tries to calculate the 379 utility terms on the GPU and
hand their 21 results directly to the GPU nested-logsum calculation. Keeping
that handoff on the GPU avoids downloading a large utility table to ordinary
memory and immediately uploading it again.

### What exactly is a skim cube?

A **skim** is a lookup table containing travel time, distance, cost, or another
measure between two zones. Imagine a map grid: choose an origin row and a
destination column, and the cell tells you the travel time. A model needs many
such maps for different travel modes and times of day. Stacking the maps makes
a three-dimensional **skim cube**.

For every possible trip, the kernel receives the already-checked origin,
destination, and time positions. It uses them like coordinates to read the
needed cells. At small scale this avoids building a giant worksheet and is
faster. At large scale, millions of rows may ask for far-apart cells. Those
scattered reads are like repeatedly fetching books from unrelated library
shelves: the arithmetic workers spend too much time waiting for data.

The next design should group nearby requests into **tiles**, reuse a fetched
skim value when several utility terms need it, and arrange memory reads so
neighboring GPU workers fetch neighboring data. This is a data-movement
problem, not evidence that the equations are too difficult for a GPU.

## 22. Three different meanings of "the answers match"

The word **exact** can describe different boundaries. A responsible report
must say which boundary it tested.

| Level | Plain-English question | Phase 15 result |
|---|---|---|
| Arithmetic | Did the strict CPU and generated GPU follow the same numeric recipe bit for bit? | Yes: all 32,262,754 checked feature cells and 1,787,646 utility cells matched exactly. |
| Decisions | Did the model choose the same alternatives? | Yes: every modeled trip decision matched in the reported runs. |
| Saved files | Were complete output files identical byte for byte? | All six non-trip substantive files were byte-identical. Every modeled field in `final_trips.csv` matched. The reported `destination_logsum` diagnostic was bounded, not byte-identical. |

The small `destination_logsum` difference is between current Sharrow arithmetic
and the newly published strict arithmetic. It was at most 0.000008 for 1,001
households and 0.00001 for 50,000 households, below the 0.0001 gate. It is not
a disagreement between the strict CPU and GPU, which matched bit for bit.

This distinction also explains why Phase 11 has the strongest production
replication statement. In its repeated 50,000-household trials, all substantive
final CSV files were byte-identical. Phase 15 has exact strict cross-device
arithmetic and exact decisions, but one diagnostic column intentionally exposes
the known difference from Sharrow's arithmetic order.

## 23. The rule for promoting an experiment

One fast run can be luck. The computer may be warmer, another program may have
used a processor core, or a file may already be cached. ChoiceForge therefore
uses fresh processes and alternates conditions: A1, B1, A2, B2, A3, B3. A is
the established implementation and B is the candidate. The **median** is the
middle of three timings, so one unusually fast or slow run has less influence.

For automatic promotion, all of these conditions must pass:

1. Run at least three fresh, interleaved A/B pairs.
2. Preserve every modeled decision and pass the complete-output correctness gate.
3. Make the median trip-destination time faster.
4. Make the median complete-model time faster.
5. Achieve complete separation: every candidate time must beat every corresponding baseline time.

| Phase 15 qualification gate at 1,001 households | Result |
|---|---|
| Three fresh interleaved pairs | Pass |
| Exact modeled decisions | Pass |
| Faster destination median | Pass: 11.3 to 10.3 seconds |
| Every candidate destination run faster | Pass |
| Faster complete-model median | Fail: 84.631 to 85.445 seconds |
| Every candidate complete-model run faster | Fail |
| Final promotion | **Rejected** |

The 50,000-household test used one pair. One pair cannot prove superiority,
but it can reveal that a candidate is clearly unpromising: Phase 15 was slower
at both the destination and complete-model boundaries. Repeating that losing
configuration three more times would spend hours without fixing its identified
memory-access problem. A redesigned kernel must return to the full three-pair
gate before any new speed claim is made.

## 24. Hardware, memory, and the boundary of the claim

The numbers in this guide describe one measured system, not every computer.

| Reproduction item | Recorded value |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 5965WX, 24 cores and 48 threads |
| System memory | 63.9 GB usable for the benchmark workstation |
| GPU | NVIDIA RTX A4000, 16 GB, compute capability 8.6 |
| NVIDIA driver | 571.59 |
| Python | 3.11.14, 64-bit |
| ActivitySim source | Commit `16ab11180a26912987eb902daf945e268f3efc11` |
| Public geography | Prototype MTC Extended, 1,454 zones |

The 50,000-household Phase 15 candidate reached 9,196 MiB of GPU memory. It did
not run out of the A4000's 16 GB. The observed slowdown instead points to the
one-block-per-row design and scattered skim reads. Another GPU with different
memory bandwidth, cache sizes, or scheduling could behave differently, so the
claim is deliberately limited to this hardware and software stack until other
systems reproduce it.

The full public population contains 2,875,192 households. The upstream
multi-process instructions call for hundreds of gigabytes of system memory,
far beyond this workstation. A 50,000-household sample with all 1,454 zones is
therefore the local public scale gate. It is large enough to expose the failed
Phase 15 access pattern, but it is not a substitute for a full-population run
on a high-memory server.

## 25. A practical reproduction checklist

Reproduction means more than downloading the code and seeing a program finish.
Another researcher should be able to identify the same source, data,
configuration, hardware, commands, outputs, and decision rule.

### Pinned evidence

| Item | Identifier recorded by the run |
|---|---|
| ChoiceForge Phase 15 evidence commit | `7f40ce227e46a16801acd6abc264544457c97a7b` |
| ActivitySim commit | `16ab11180a26912987eb902daf945e268f3efc11` |
| Prototype MTC `data_full.tar.zst` SHA-256 | `b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3` |
| ActivitySim integration patch SHA-256 | `aa2eed95d99fe4853365b9ed49427722ac467ef928013ccf7944fd9eda0e04e2` |
| Python environment lock SHA-256 | `84b97738100cd4b8c405af50ff823cdddf4a33f5f203deab7839e4a8739a3adc` |

### Set up and verify the software

The detailed setup is in `docs/activitysim-integration.md`. In PowerShell, the
important verification steps are:

```powershell
git clone https://github.com/ActivitySim/activitysim.git tmp\activitysim-phase8-source
git -C tmp\activitysim-phase8-source checkout 16ab11180a26912987eb902daf945e268f3efc11
git -C tmp\activitysim-phase8-source apply ..\..\integration\activitysim-current-choiceforge.patch
uv venv --python 3.11 .venv-phase8
uv pip install --python .venv-phase8\Scripts\python.exe -e tmp\activitysim-phase8-source -e ".[gpu,test]"
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

Place the public full-geography data at
`benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full`. Check its
downloaded archive against the SHA-256 value above before extracting it. The
benchmark scripts refuse to overwrite an existing run, so use a new short
`RunTag` when repeating an experiment.

### Run the exactness and performance gates

First run one qualification candidate. Then run three direct Phase 11 versus
Phase 15 pairs:

```powershell
.\scripts\run_phase15_candidate.ps1 -Households 1001 -RunTag reproduce-gate -MaxCandidateRows 2000000
.\scripts\run_phase15_incremental_ab.ps1 -Households 1001 -Repetitions 3 -RunTag reproduce-r3 -MaxCandidateRows 2000000
```

The candidate is explicitly enabled with
`CHOICEFORGE_STRICT_CUDA_CANDIDATE=1`; it is off by default. The row policy
`CHOICEFORGE_STRICT_CUDA_MAX_ROWS` can reject an ineligible batch before a
large allocation. Any code-generation or GPU-reduction failure falls back to
Sharrow, which remains authoritative.

Compare a new run with these machine-readable evidence files:

- `benchmark-results/phase15-candidate-summary.json`
- `benchmark-results/phase15-p15finalr3-summary.json`
- `benchmark-results/phase15-p15compact50-summary.json`

The manifests record source, patch, environment, data and configuration hashes,
GPU and driver, Python version, individual timing samples, memory measurements,
correctness results, and the final promotion decision. Matching the published
median alone is not enough; the correctness and promotion fields must also pass.

## 26. The Phase 16 plan that was tested

Phase 15 showed that the next major opportunity was not to add more arithmetic
to the existing one-block-per-row kernel. The approved Phase 16 plan was to
change how data reaches that arithmetic and test each idea honestly:

1. Add measurement for skim-cache hits, memory transactions, and time spent waiting on gathers.
2. Sort or tile rows by origin, destination, and time without changing their final model order.
3. Coalesce neighboring skim reads and reuse one gathered value across every term that needs it.
4. Fuse tiled gathering, strict ordered utility accumulation, and nested-logsum reduction without writing a global feature or utility matrix.
5. Preserve strict IR version 3, typed inputs, default-off activation, bounded diagnostics, and Sharrow fallback.
6. Pass unit tests and exact CPU/GPU edge cases before running ActivitySim.
7. Qualify all 30 public-model batches and exact final decisions at 1,001 households.
8. Require three interleaved 50,000-household pairs and the full promotion rule before replacing Phase 11.
9. Repeat on at least one different GPU and one differently structured public ActivitySim model.
10. Only then propose the backend for upstream Sharrow or ActivitySim integration; until every gate passes, Phase 11 remains supported and Phase 15 remains opt-in research evidence.

## 27. Phase 16: the GPU finally wins the large target component

Phase 15 taught us that moving a calculation to a GPU does not automatically
make it faster. Phase 16 used that failure as a map. It measured where time was
going, tried several designs, rejected the slow ones, and found one policy that
was both reproducible and faster for the targeted part of the public model.

### Four ideas in ordinary language

First, constants no longer become giant repeated columns. If every row uses the
same number, the GPU receives that number once. This is like writing a classroom
rule on the board instead of printing the same rule on every student's paper.

Second, two names that point to the exact same column share one stored copy.
Third, travel-time maps and compiled equation plans are cached so the program
does not rediscover identical information for every batch. Fourth, an optional
policy uses 32-bit floating-point arithmetic for the expressions instead of
64-bit arithmetic.

The last change matters on this NVIDIA RTX A4000. It can do ordinary 32-bit GPU
arithmetic much faster than 64-bit arithmetic. Sharrow already stores each
completed feature as a 32-bit number before combining features with
coefficients, so Phase 16 publishes a separate 32-bit recipe and a separate CPU
answer key for it. The older strict 64-bit recipe is still available and is not
silently changed.

### What is a numeric policy?

A **numeric policy** is a written recipe for how the computer handles numbers.
It says how many bits a number uses, when rounding happens, whether operations
may be rearranged, and how unusual values such as infinity are treated. Two
programs can use the same equation but get slightly different last digits if
their numeric policies differ.

ChoiceForge now has two named policies:

| Policy | Expression arithmetic | Why it exists |
|---|---|---|
| Strict IR version 3 | 64-bit, then one 32-bit feature rounding | strongest continuation of the Phase 13-15 arithmetic contract |
| Phase 16 FP32 | 32-bit expression arithmetic and 32-bit feature storage | mirrors the deployed intermediate precision more closely and runs efficiently on this GPU |

The run manifest records which policy was selected. The generated source hash
also changes, so evidence from the two policies cannot be accidentally mixed.

### Did the new CPU and GPU answers match?

Yes. On the real 1,001-household qualification:

| Exact qualification question | Result |
|---|---:|
| Real batches checked | 30 of 30 |
| Rows checked | 85,126 |
| Feature cells exactly equal | 32,262,754 of 32,262,754 |
| Utility cells exactly equal | 1,787,646 of 1,787,646 |
| Largest CPU/GPU difference | 0.0 |
| Fallback batches | 0 |

The tests also include a simple equation where the 64-bit answer is 1 and the
32-bit answer is 0 because a very large 32-bit number cannot represent a change
of just 1. That test proves the two policies really are different. It then
proves that the published 32-bit CPU and GPU recipes agree exactly.

### The large repeated performance proof

The public scale gate uses 50,000 households, all 1,454 zones, and 4,188,312
utility rows. It launches fresh processes in the order baseline 1, candidate 1,
baseline 2, candidate 2, baseline 3, candidate 3.

| Trip destination | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Phase 11 baseline | 28.5 s | 28.5 s | 28.8 s | 28.5 s |
| Phase 16 FP32 GPU | 28.4 s | 27.7 s | 27.8 s | 27.8 s |

Every candidate time beat every baseline time. The median speedup is 1.025
times, and the paired savings are 0.1, 0.8, and 1.0 seconds. Every modeled trip
decision matched in all three candidate runs. The largest printed diagnostic
logsum difference was 0.000010, below the 0.0001 limit.

This means the **component promotion gate passes**. It is a practical,
repeatable GPU win for the part of the model Phase 16 was designed to replace.

### Why is the whole model not promoted?

The full model contains many steps Phase 16 does not change. Their timings move
slightly from run to run because of operating-system scheduling, file caches,
CPU activity, and other noise. Whole-model medians were:

| Complete model | Median |
|---|---:|
| Phase 11 baseline | 193.014 s |
| Phase 16 candidate | 194.312 s |

That is a 0.993-times result, which is slower, not faster. One candidate run was
faster than its paired baseline, but two were slower. The **whole-model gate
fails**. Reporting both gates prevents a real kernel success from being hidden
and prevents that success from being exaggerated into an application-wide
claim.

### Slow ideas that were tested and rejected

Phase 16 did not simply keep the first plausible code:

- Loading 149 skim maps cooperatively for tiles of nearby rows was exact but
  synchronization and unnecessary eager loads made it slower.
- The model has only about 419-465 nonzero coefficients out of 7,959 positions.
  Skipping zeros sounds ideal, but one version created bulky control flow and
  another caused scattered shared-memory reads. Both were slower.
- Strict FP64 compaction reduced host packing from about 2.4 to 1.1 seconds and
  utility time from about 4.5 to 4.1 seconds, but destination still lost 32.5
  to 30.1 seconds at large scale.
- FP32 reduced the large generated utility work to roughly 1.2-1.6 seconds and
  produced the repeated component win.

The lesson is that counting arithmetic operations is not enough. GPUs care
about memory order, groups of threads following the same instruction, code
size, and the kinds of number hardware they execute efficiently.

### What Phase 17 needed to do

Phase 17 had three jobs: stop rebuilding the same GPU program, prevent that
optimization from moving a delayed Sharrow cost into trip mode choice, and
repeat the public benchmark enough times to separate a real destination win
from ordinary timing noise. The next section explains what was implemented and
what the measurements do and do not prove.

## 28. Phase 17: turn a kernel into a reusable backend

A fast kernel is like a fast kitchen appliance. It is not enough for the
blender blades to spin quickly if someone must unpack, assemble, and wash the
whole machine for every smoothie. Before Phase 17, ChoiceForge repeatedly did
small pieces of setup around an already-fast generated kernel.

### A compiled plan is a checked instruction packet

Phase 17 creates a **compiled plan**. The plan remembers:

- the exact equation recipe, identified by a cryptographic hash;
- which inputs are whole columns and which are single numbers;
- the data type and storage slot for every input;
- how skim lookups are indexed;
- the compiled CUDA kernel; and
- the coefficient matrix already stored on the GPU.

Reusing a plan is safe only when the next batch has the same shape of meaning.
The row values may change. A price or coefficient value may change. But a
column cannot quietly become a scalar, a floating-point value cannot quietly
become an integer, and two different columns cannot quietly acquire a new
shared-memory layout. If any of those rules changes, ChoiceForge refuses that
plan and builds or selects a matching one. This is called **fail closed**: when
the program is uncertain, it does not guess.

There is a subtle scalar rule too. Imagine two constants both happen to equal
2 today. Combining them into one slot would be wrong if tomorrow one becomes
3 and the other becomes 7. Persistent mode therefore gives semantic scalar
inputs stable slots even when their current values are equal.

### Why trip mode choice had to join the GPU path

An early experiment made destination faster but made the later trip-mode step
about 2.4 seconds slower. The code had not created work from nowhere. It had
stopped Sharrow from warming up during destination, so Sharrow paid its first
compilation cost later during mode choice. This is a **displaced cost**: one
timer looks better because the cost moved to another timer.

Phase 17 adds a continuation bridge. The same generated FP32 CUDA utility
plans now serve trip mode choice. ActivitySim still owns the nested-logit
probabilities, random numbers, final choices, and result tables. ChoiceForge
replaces only the utility calculation. If the generated path encounters an
unsupported case, it records the reason and calls the original Sharrow path.
The reported runs had zero fallbacks.

### The small exactness gate

On 1,001 public households, the final Phase 17 qualification checked:

| Check | Result |
|---|---:|
| Real destination batches | 30 of 30 exact |
| Rows evaluated | 85,126 |
| Feature values | 32,262,754 of 32,262,754 exact |
| Utility values | 1,787,646 of 1,787,646 exact |
| Reused destination plans | 20 calls |
| Reused trip-mode plans | 10 calls |
| GPU fallbacks | 0 |
| Changed modeled decisions | 0 |

Two saved columns are diagnostics rather than later decision inputs. Their
last decimal places can differ because the declared FP32 GPU policy does not
promise identical decimal text to Sharrow:

| Diagnostic | Largest difference | Allowed gate |
|---|---:|---:|
| Destination logsum | 0.000008 | 0.0001 |
| Mode-choice logsum | 0.00000191 | 0.00001 |

The output checker gives tolerance only to those two named columns. Every trip
destination, trip mode, and other modeled output still has to match exactly.

### The five-pair 50,000-household result

The clean proof used five fresh baseline/candidate pairs and fixed both BLAS
thread settings at 16. A prior 24-thread attempt suffered a native OpenBLAS
failure in a baseline process, so that incomplete series was discarded.

| Measure | Baseline median | Phase 17 median | Result |
|---|---:|---:|---:|
| Trip destination | 28.4 s | 27.3 s | 1.040x faster |
| Complete model | 191.474 s | 190.307 s | 1.006x faster |

Every one of the five destination runs was faster than every baseline
destination run. The five paired destination savings were 1.8, 0.8, 0.9, 0.8,
and 1.9 seconds. A bootstrap calculation put the middle saving's 95% interval
between 0.8 and 1.9 seconds. That is strong component-level evidence.

The complete model is more complicated. Its paired savings were 1.845,
-0.042, 1.360, -0.038, and 2.595 seconds. Three won and two lost by less than
one twentieth of a second. The median improved by 1.167 seconds, but the 95%
bootstrap interval ran from -0.042 to 2.595 seconds. Because that interval
still touches a small slowdown, the strict whole-model superiority gate does
not pass. “The median was faster” is true; “whole-model superiority is proven”
would be too strong.

### Reusable buffers: useful, but still experimental

Creating GPU arrays also costs time. Phase 17 implements an optional workspace
that lets a checked plan reuse its device input and output storage. A first
version used pinned host memory. It uploaded faster but packed columns more
slowly, so it was rejected. The retained version keeps fast NumPy column
packing and reuses only device allocations.

That version passed the exact gate. In one 50,000-household diagnostic it
reduced measured destination packing plus upload time from about 1,471.5 to
1,364.1 milliseconds, a saving of roughly 107 milliseconds. However, one pair
cannot prove a whole-model effect, and unrelated initialization varied by more
than two seconds in that run. Reusable buffers therefore remain an opt-in
experiment, not part of the five-pair headline proof.

## 29. What the result means and what comes next

Here is the honest Phase 17 conclusion:

- the generated FP32 GPU backend exactly matches its published FP32 CPU answer
  key on every qualified feature and utility cell;
- all reported modeled decisions match the reference;
- repeated trip-destination superiority is proven on this RTX A4000 and public
  MTC workload;
- the five-run whole-model median is faster, but the strict whole-model proof
  is not yet complete; and
- the result has not yet been replicated on a second GPU or a different public
  ActivitySim model.

The fastest path to a stronger claim is to collect more independent,
controlled 50,000-household pairs with the same locked software, hashes, and
16-thread setting. That narrows the uncertainty around the tiny whole-model
losses. The most important scientific step after that is external replication:
run the same exactness and A/B protocol on a second NVIDIA GPU and then on a
second public ActivitySim model. A result that survives different hardware and
different travel equations is much more useful to the ActivitySim and Sharrow
communities than a single impressive number from one workstation.

## 30. Phase 18: can a chain of model steps live on the GPU?

After Phase 17, the obvious ambitious question was: what if the GPU did not
calculate one isolated piece and then hand everything back to the CPU? What if
household data entered the GPU once, several dependent model steps happened
there, and only the finished answers came back?

Phase 18 builds the first honest version of that idea. It is not a whole travel
model yet. It is a **vertical slice**, meaning a narrow but complete path from
real input rows through multiple calculations to final outputs. Imagine
building one fully working elevator before constructing every floor of a
skyscraper. The elevator proves the important machinery can work together, but
it does not mean the building is finished.

### Does “GPU-only” really mean the CPU does absolutely nothing?

No. A regular computer must use its CPU to start Python, read a CSV file, read
configuration, tell the GPU which kernel to launch, handle errors, and write a
result file. Even a CUDA program is launched by host code.

In this project, **GPU-native modeled execution** has a precise meaning:

- The CPU may read input and upload one partition before modeling starts.
- The CPU may launch kernels and wait for them.
- The CPU may download final outputs after modeling finishes.
- The CPU may not calculate a utility, random choice, logsum, or modeled total.
- The program may not secretly use NumPy or pandas when a GPU operation is
  missing.

The runtime closes a gate called `seal_ingress`. After that gate closes, a
host array entering a modeled stage, an intermediate modeled result leaving
the GPU, or a CPU fallback causes a hard error. This behavior is called
**fail closed**. If the program cannot honor its promise, it stops instead of
quietly producing a misleading benchmark.

### What is GPU state?

A travel model changes tables over time. A household first gets an auto
ownership result. A person later gets a work location. Tours and trips are then
created. Later steps depend on earlier answers. All those current tables are
the model's **state**.

Phase 18 adds a `DeviceTable`. Its columns must be CUDA arrays, which means the
actual numbers live in GPU memory. All columns must describe the same number of
rows. A `GpuNativeRuntime` owns these tables and records what happens at the
CPU/GPU boundary. It counts input bytes, output bytes, modeled transfers, CPU
fallbacks, and kernel stages.

### Why random numbers are part of correctness

Travel models use random draws to turn probabilities into simulated choices.
Suppose household 42 gets random number 0.63 in one run. If changing the batch
size gives it 0.18, the model can change even though no travel assumption
changed. That would make scaling unsafe.

The new GPU random generator uses three stable labels:

```text
entity ID + project seed + stream ID -> the same random draw
```

The entity ID identifies the household or person. The project seed identifies
the run. The stream ID identifies the decision, such as first choice or second
choice. A hash thoroughly mixes those integers, and the GPU turns part of the
result into a number between zero and one.

The important fact is what the formula does **not** use: it does not use the
row's position inside a partition. Household 42 therefore gets the same bits
whether it is processed in one table of 2.8 million households or a small table
of 250,000. A separate NumPy implementation checks the GPU generator bit for
bit.

### Why adding numbers can be nondeterministic

Floating-point addition is not perfectly associative. In exact mathematics,
`(a + b) + c` equals `a + (b + c)`. On a computer, rounding after each addition
can make the last bits different.

Many fast GPU totals use **atomics**: many threads race safely to add their
values to one total. The final answer can depend on which thread arrives first.
Phase 18 instead sorts group IDs and gives one thread responsibility for each
group. That thread adds rows in a fixed order. It may not be the fastest
possible future method, but its behavior is easy to explain and reproduce.

### What the public-data vertical slice does

The benchmark reads all 2,875,192 households from the public Prototype MTC
table. It then performs this chain after the GPU boundary closes:

- **Stage 1:** Build eight household features, such as normalized household size, workers, automobile count, income, and zone information.
- **Stage 2:** Generate the first stable random stream on the GPU.
- **Stage 3:** Make a fused choice among 21 alternatives.
- **Stage 4:** Put that first answer into the inputs of a second choice.
- **Stage 5:** Generate a different stable stream and make the dependent second choice.
- **Stage 6:** Sort households by traffic analysis zone, or TAZ, and total the first choices within each zone.
- **Stage 7:** Download the final answers for checking.

The word **dependent** matters. The second choice really uses the first choice.
This is not a collection of unrelated kernels that happen to run one after
another.

### Which parts are real, and which are invented for the test?

The households and their fields come from the real public MTC benchmark. The
21 alternatives and eight-feature calculation have the shape of travel-model
choice work. However, the coefficients are fixed synthetic test values. Nobody
estimated them from a survey, so these choices must never be interpreted as a
forecast of real behavior.

The data transform also publishes its assumptions:

- Negative income codes mean missing for this systems test and become zero.
- Income above $250,000 is capped before normalization.
- Very large household, worker, and automobile counts are capped before
  normalization so rare coding outliers do not dominate the test.
- Household ID is the stable random key.
- The calculation uses a declared 32-bit floating-point policy.

These choices make a stable systems benchmark. They do not claim to be the
right behavioral specification for a transportation agency.

## 31. Phase 18 results: fast, reproducible, and carefully bounded

The full-table benchmark used the local NVIDIA RTX A4000, driver 571.59,
Python 3.11.14, nine measured repetitions, and a fused parallel Numba CPU
comparison. Compilation warm-up happened before the measured repetitions.

| Full public household table | Median time |
|---|---:|
| Parallel fused Numba CPU | 0.457023 seconds |
| GPU modeled work only | 0.031357 seconds |
| GPU including one upload and final downloads | 0.055327 seconds |

That is **14.575 times faster** for modeled computation and **8.260 times
faster** even after the allowed boundary transfers.

This is a much larger speedup than the whole ActivitySim results reported in
earlier phases because the scopes are different. Phase 18 measures a compact
GPU-native vertical slice where almost every operation is parallel. It does not
include CSV parsing, all ActivitySim components, checkpoint work, or output
writing. A person comparing the numbers must keep that difference in mind.

### What exactly reproduced?

The same GPU calculation ran in two arrangements:

- all 2,875,192 households together; and
- consecutive partitions containing at most 250,000 households each.

Every final GPU choice was identical. Every final GPU logsum had identical
bits. This is the Phase 18 **replication guarantee**: scaling the workload into
deterministic partitions does not change the GPU result.

Runtime telemetry also showed:

| Boundary check | Result |
|---|---:|
| Modeled CPU fallbacks | 0 |
| Host-to-device modeled bytes after ingress closed | 0 |
| Device-to-host modeled bytes before final output | 0 |
| Sampled active GPU allocation peak | about 452.4 MiB |

The memory number samples active CuPy arrays after stages. A very short-lived
temporary allocation could be higher, so it is a measured lower bound rather
than a perfect hardware high-water mark.

### Why are CPU and GPU answers not all bit-identical here?

The Numba CPU and CUDA GPU both implement the same mathematical model, but they
do not promise the same lowest-level versions of exponential and
fused-multiply-add operations. Across all 2,875,192 rows:

| CPU/GPU comparison | Observed difference |
|---|---:|
| First choices | 1 row |
| Dependent second choices | 3 rows |
| Largest dependent logsum difference | 0.007143 |
| Largest zone-total difference | 1.0 |

One first-stage boundary choice can affect the second-stage feature, which is
why the number grows from one to three. The observed rates are 0.348 and 1.043
differences per million rows. They pass the published limits of one and two per
million.

This is not the same guarantee as Phases 14-17, where a specially defined CPU
recipe and generated CUDA utility recipe matched feature and utility bits
exactly. Phase 18 deliberately compares against a strong normal Numba choice
implementation. Its cross-architecture claim is **numerical equivalence within
written bounds**, while its GPU partition claim is **bit-exact**.

During development, a first full-table run used `log1p(income)`. NumPy and CUDA
rounded that function differently by one float32 unit on some rows, and one
row landed close enough to a choice boundary to change. The project did not
hide the failure or simply rerun until it disappeared. It replaced the feature
with capped linear normalization, documented the arithmetic policy, reran the
full table, and still reported the remaining CUDA-versus-Numba boundary cases.

### What fits in this GPU's memory?

The local RTX A4000 reports 16,376 MiB of GPU memory. The public MTC skim file
contains 826 datasets. If every dataset is counted as its raw uncompressed
array, the collection needs 13.389 GiB. Earlier real 50,000-household
ActivitySim integration runs already peaked near 8.4 GiB.

Therefore the complete future model should not load every skim and every state
table at once. Phase 18 proposes four planned pools:

| Planned use | Budget |
|---|---:|
| CUDA, driver, and failure reserve | 2 GiB |
| Frequently used, or “hot,” skims | 4 GiB |
| Persistent model state | 2 GiB |
| Largest component workspace | 3 GiB |
| Remaining safety and partition room | about 4.99 GiB |

A **hot skim cache** keeps the travel-time maps needed soonest on the GPU and
evicts others safely when space is needed. A **population partition** processes
a stable set of households and everything belonging to them, then moves to the
next set. These are design budgets, not yet measured full-model limits. The
maximum safe production partition cannot be claimed until the missing model
components have real memory high-water measurements.

### What does Phase 18 prove?

- A sealed GPU state boundary can be enforced rather than merely promised.
- Stable GPU random draws survive changes in partition size.
- Dependent choices and an ordered group total can stay on the device.
- This vertical slice can process the entire public household table at once.
- The GPU is substantially faster than fused parallel Numba for this work, including the boundary transfers.
- Every declared performance, numerical, partition, and fallback gate passes.

### What does it not prove?

- It is not a complete ActivitySim replacement.
- Its synthetic coefficients do not make calibrated travel predictions.
- It does not prove every person, tour, trip, timetable, shadow-price, and skim table fits together in 16 GB.
- It does not yet provide restartable GPU checkpoints.
- It does not accelerate file parsing or report writing.
- It does not prove the same speed on another GPU or model.
- It does not claim that ordinary CPU and GPU math is bit-identical.

## 32. The practical path from this slice to a whole GPU model

The next-generation project is possible on this GPU if it grows in dependency
order and treats memory and replication as design rules.

### Phase A: finish the GPU table toolbox

Implement indexed joins, filters, stable sorting, scatter and gather, category
encoding, group operations, and missing-value policies. Each operation needs a
small readable CPU oracle, adversarial tests, and an explicit arithmetic rule.

### Phase B: build the hot skim cache

Track which skim arrays each component needs. Upload them asynchronously,
retain frequently reused arrays, and evict only after every dependent CUDA
event finishes. Record cache hits, misses, transferred bytes, and memory peaks.

### Phase C: port one calibrated household-person chain

Replace synthetic coefficients with a real public specification. Preserve
stable entity channels and compare every intermediate column, not only the
final output. A missing GPU operation must stop qualification rather than use
a hidden CPU fallback.

### Phase D: make restart behavior safe

Long models must recover after a failure. Either serialize device tables at
declared checkpoints or make a documented host checkpoint boundary. Restoring
a checkpoint must reproduce random streams and every downstream table.

### Phase E: add tours, trips, timetables, destinations, and shadow prices

These are harder because they create variable numbers of rows, use large skim
lookups, and update shared state. Port them one dependency layer at a time.
Measure high-water GPU memory after every component and reduce partition size
before an out-of-memory failure becomes possible.

### Phase F: run the real whole-model proof

Only after the calibrated chain is complete should the project run fresh,
interleaved CPU/GPU whole-model trials. The proof must include table hashes,
random-stream checks, fallback counters, memory peaks, transfer bytes, component
times, whole-model time, software hashes, and a second-machine replication.

The implication is exciting but specific. This workstation has enough GPU to
build and test a serious next-generation model and to run large partitions very
quickly. It does not have enough memory to hold the public model's entire skim
collection plus every future state object without caching and partitioning.
Success therefore comes from a GPU-native **system** - state, randomness,
memory, checkpoints, and kernels together - not from writing one spectacular
kernel.

## 33. Phase 19: replace the invented equations with a real calibrated chain

Phase 18 answered an engineering question: can several dependent calculations
stay on the GPU and run quickly? Its answer was yes, but its choice equations
used invented test coefficients. Phase 19 asks the harder scientific question:

> Can the GPU run real published travel-model equations and reproduce the
> choices that ActivitySim already made?

The answer for the first calibrated household-to-person chain is **yes**.

### What does “calibrated” mean?

A travel-model equation combines facts such as income, household size, age,
location, and travel time. Each fact gets a number called a **coefficient**.
A coefficient says how strongly that fact moves a choice up or down.

For example, this made-up equation is only an illustration:

```text
score for owning two cars
  = 0.4 × number of workers
  + 0.2 × income in thousands
  - 0.3 × transit accessibility
```

Real modelers estimate coefficients from observed travel and household data.
When a model uses those estimated numbers, it is **calibrated**. Phase 19 uses
the real coefficients published with Prototype MTC Extended. It does not invent
new behavior.

### Which two real decisions run on the GPU?

The first model is **auto ownership**. Each household chooses one of five
answers: zero, one, two, three, or four-or-more automobiles.

The second model is **mandatory tour frequency**. A *tour* is a journey that
starts at home, visits one or more places, and returns home. A mandatory tour
goes to work or school. A person can choose among these five answers:

- one work tour;
- two work tours;
- one school tour;
- two school tours; or
- both a work and a school tour.

Only people whose earlier daily-activity choice says they have a mandatory
activity enter the second model. In the public checkpoint, that is 78,900 of
132,536 people.

The chain is truly connected:

```text
50,000 households
        |
        v
GPU auto-ownership model
        |
        | join each answer to people in that household
        v
78,900 mandatory-person choosers
        |
        v
GPU mandatory-tour-frequency model
```

The second model uses the **new GPU auto answer**. It does not secretly copy
ActivitySim's saved auto answer into the calculation. The saved answer is used
only afterward as an answer key.

### What is a checkpoint replay?

A long video game lets you save at a checkpoint instead of starting from the
beginning after every test. ActivitySim can also save its tables after model
steps. Phase 19 starts from public saved tables immediately before the chosen
components.

This is powerful because the input state is fixed. The CPU reference, the GPU,
and ActivitySim's saved output all begin from the same households, people,
zones, work locations, school locations, accessibility values, and earlier
daily-activity answers.

It also limits the claim. Phase 19 does **not** recalculate school location,
work location, or the earlier daily-activity model on the GPU. Those are frozen
inputs. This is a two-component calibrated replay, not yet a whole-model run.

## 34. How Phase 19 makes a real choice

Both components use a method called **multinomial logit**, or MNL. The name is
less important than its four steps.

### Step 1: evaluate expressions

An **expression** turns input columns into a useful model feature. Examples in
the real auto-ownership file include:

- whether the household has exactly two drivers;
- household income, capped within a range;
- retail accessibility by car or transit;
- whether the home is in San Francisco County; and
- estimated automobile time savings per worker.

The auto model has 29 active expressions. The mandatory-tour model has 98.
Together, the GPU evaluates 127 published expressions.

The safe expression reader understands only operations that were reviewed and
tested, such as addition, comparison, Boolean “and/or,” clipping a value to a
range, and choosing one of two values with `where`. If an unfamiliar operation
appears, the run stops. It never sends the expression to unrestricted Python
code and never quietly falls back to the CPU.

### Step 2: calculate utilities

For every alternative, each expression value is multiplied by its calibrated
coefficient. The products are added to form a **utility**. Utility is a model
score, not money or electrical service. A higher utility means the alternative
is more attractive to the model.

With 50,000 households and five auto alternatives, the first component makes
250,000 utility values. With 78,900 people and five tour-frequency alternatives,
the second makes 394,500 utility values.

### Step 3: turn utilities into probabilities

MNL converts the five utilities in a row into five probabilities that add to
one. A numerically safe version first subtracts the largest utility, then uses
the exponential function, and finally divides each result by the row total.

Suppose the probabilities were:

```text
alternative:  A     B     C     D     E
probability: 0.10  0.25  0.40  0.20  0.05
```

These probabilities describe ranges on a line from zero to one. A random draw
of 0.52 lands in C's range, so C is selected.

A **logsum** is the logarithm of the total exponentiated utility. Modelers use
it as a summary of how attractive the complete choice set is. Phase 19 keeps
logsums on the GPU and downloads them only as final diagnostic outputs.

### Step 4: use ActivitySim's exact random draw

Matching probabilities is not enough. Two programs can have the same
probabilities but choose different answers if they use different random draws.

ActivitySim creates a stable seed from four items:

```text
run seed + channel name + model-step name + household/person ID
```

It then uses an algorithm called **MT19937**, from NumPy's older
`RandomState`, to create the draw. Phase 19 implements the needed part of
MT19937 directly in a CUDA kernel. The GPU produces the exact same 64-bit
floating-point draw as NumPy for every tested household and person.

This kernel currently supports the first draw in a model step, called offset
zero. These two public components need exactly that. If a future component asks
for a later draw, the system stops until that behavior is implemented and
proved.

### How does a person find the right household answer?

Tables do not always have rows in matching order. A person's row contains a
`household_id`. The GPU must find the household row with that ID.

Phase 19 adds a GPU **keyed join**. It sorts source IDs, searches for every
requested ID, checks that no ID is missing, and gathers the matching values.
The same tool also looks up land-use and accessibility facts by zone ID.

This is a basic database operation performed on the GPU. It is essential for a
whole travel model because household, person, tour, trip, and zone tables
constantly refer to one another by IDs.

## 35. How we proved the result and what it means

The proof uses three separate answer layers.

1. A plain NumPy version independently calculates expressions, utilities,
   probabilities, ActivitySim random draws, and choices.
2. That CPU version must exactly reproduce ActivitySim's saved checkpoint
   choices.
3. The GPU is compared with every important CPU intermediate and separately
   with ActivitySim's saved final choices.

This avoids circular reasoning. The GPU is not declared correct merely because
another function that shares all its code agrees with it.

### Correctness results

| Public-checkpoint test | Result |
|---|---:|
| Households | 50,000 |
| All people in input state | 132,536 |
| Mandatory-person choosers | 78,900 |
| CPU auto choices different from ActivitySim | **0** |
| GPU auto choices different from ActivitySim | **0** |
| CPU tour-frequency choices different from ActivitySim | **0** |
| GPU tour-frequency choices different from ActivitySim | **0** |
| Expression feature differences | **0** |
| Random-draw bit differences | **0** |
| Largest utility difference | about `0.0000000000000018` |
| Largest probability difference | about `0.00000000000000044` |
| Choice differences across nine repeated GPU runs | **0** |

Those tiny utility and probability differences come from low-level
floating-point library arithmetic. They are many orders of magnitude below
the written limits and did not change a single choice.

### Speed results

The program first warmed up compilation. It then measured nine complete CPU
and GPU replays on the RTX A4000.

| Measured boundary | Median time | Relative speed |
|---|---:|---:|
| Independent CPU replay | 0.458724 seconds | 1.000× |
| GPU modeled work | 0.025713 seconds | **17.840× faster** |
| GPU with input upload and final download | 0.037997 seconds | **12.073× faster** |

The transfer-inclusive number is the most practical result for this checkpoint
replay. It includes moving the input tables to the GPU once and bringing the
final choices and logsums back. It does not include reading Parquet files from
disk or running all the upstream and downstream ActivitySim components.

### Did the GPU secretly use the CPU?

No modeled fallback was recorded. After the GPU boundary closed:

| Forbidden event | Recorded amount |
|---|---:|
| CPU modeled fallbacks | 0 |
| Late host-to-GPU modeled transfers | 0 bytes |
| Early GPU-to-host modeled transfers | 0 bytes |

The CPU still launches kernels and checks a few true/false error flags. That is
control work, not a second model calculation.

### Does Phase 19 make the first 18 phases useless?

No. It makes Phase 19 the best **headline**, but it could not stand alone as a
trustworthy project.

Earlier phases built and tested the choice kernels, strict expression language,
float rules, ActivitySim adapters, public checkpoints, failure policies,
device-state runtime, and transfer counters. They also studied destination and
scheduling components that Phase 19 does not run. Phase 18 proved that a
dependent GPU state chain could work before real calibrated behavior was added.

The right summary is:

- Phase 19 replaces Phase 18's synthetic behavioral demonstration for this
  household-to-person boundary.
- Phases 1-18 remain the engineering foundation, audit trail, and evidence for
  other component types.
- Phase 19 still does not replace a full fresh ActivitySim CPU/GPU comparison.

### What should happen next?

The next major challenge is not another fixed-size five-answer choice. It is
creating new tour rows on the GPU.

If one person chooses two work tours, the model must create two rows. If another
chooses one school tour, it creates one row. This is called **variable-length
table expansion**. It needs counts, prefix sums, stable new IDs, repeated person
fields, category codes, and deterministic ordering.

The practical Phase 20 sequence is:

1. Turn each mandatory-tour-frequency choice into the correct number and type
   of GPU tour rows.
2. Give every tour a deterministic ID and verify the complete table against
   ActivitySim.
3. Run one downstream tour model that uses those rows and travel-time maps.
4. Add general random-channel offsets before a component needs multiple draws.
5. Save a restart manifest containing table hashes, schemas, random offsets,
   and completed steps.
6. Repeat the chain on a larger public population and a second NVIDIA GPU.

Phase 19 is a major success because it combines real calibrated behavior,
exact public choices, a genuine household-to-person dependency, a fail-closed
GPU boundary, and a double-digit measured speedup. The remaining distance to a
whole model is mostly the hard state-management work: creating rows, managing
skims, scheduling time, checkpointing, and covering the rest of the component
graph without breaking exactness.
