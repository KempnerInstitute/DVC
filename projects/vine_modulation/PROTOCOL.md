# Interpreting the vine modulation study

## What changes in the continuous model?

A simplified vine describes each conditional pair by a copula whose parameters can change over time. A modulated edge allows its conditional association to vary with the observed value of another variable. A product of two conditioning features allows four variables to participate. The fitted coefficient and its interpretation depend on the chosen root order, conditioning set and link function. The level of an edge in the tree does not identify the order of an interaction.

The study asks three separate questions. Does a modulated edge improve held-out prediction? Does its coefficient change across windows? Can the analysis recover the affected variables and the onset and end of the episode? For exact recovery, the simulation score requires the correct subset, a change decision and the correct active windows. Close timing errors count as misses. Figure 3 gives the results by law and sample size, including the pairwise controls.

## What is measured in a ripple?

The data unit is a hippocampal sharp-wave ripple. A cell participates if it spikes in the event window. For three cells, the eight participation patterns form a binary contingency table. The conditional three-way log odds can vary while every two-cell margin stays fixed. The study estimates how this quantity changes across a track experience, comparing cells with overlapping place fields with cells whose fields are separate. Pooling across trios accounts for shared cells and events. Whole-ripple prediction is a separate analysis of the complete cell pattern.

## What limits the inference?

The first cohort's matched simulations exceeded two prespecified bounds. Before computing the recording contrasts, the analysis introduced a minimum three-way effect of 0.05 log units and excluded the animal with only four eligible trios from that question. In the second cohort, changing pairs alone produced a 10% rejection rate against a prespecified upper limit of 8%. The three-way standard errors were multiplied by 1.11 before the data contrasts were measured. The study drivers retain the initial criteria, the amendments and the simulation results. These checks qualify inference under changes in the surrounding pairs.

A before-to-after contrast describes differences between recording periods. The blind timing analysis asks whether the data also locate a discrete change. A replay model tests how much of the contrast could arise when different places are represented during ripples. Its agreement varies across cohorts. A residual relative to this model therefore depends on how well it predicts the recorded cells.

## Reproduction

The project README maps all seven figures to the scripts that generate them. Each figure can be rebuilt from the included data tables. Repeating spike extraction, field classification or replay fitting requires the source recordings and the session-level commands in that guide.
