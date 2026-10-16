# Performance

Each release is timed against the previous one on the same machine, so differences in
machine speed cancel out. Times on your own machine will differ; compare the change column,
not the seconds.

**How it is measured.** `pixi run speedtest --previous X.Y.Z` runs each model three times
under the previous release and under the new code and reports the median.

- **Chemistry**: time inside PhreeqcRM (`PhreeqcBMI._solve_phreeqcrm`).
- **Transport**: time inside MODFLOW 6 transport (`Mf6API._solve_gwt`).
- **Other**: the rest of the run (setup, output writing, data handoffs).
- **Change**: total time, new versus old; negative is faster.

Models: ex7 and ex8 from the benchmarks, and `ex5_large`, the ex5 column with more cells so
chemistry dominates. ex8 runs twice (conservative, then reactive); its total covers both.

<!-- Paste each release's table below, newest first. -->
