# Software and study releases

The library installs from the repository root. Each research project identifies the code, data products and commands used for its analysis. The vine modulation project includes the scripts and numerical inputs needed to regenerate all seven figures.

## Reproduce a specific revision

Record the Git commit of the checkout used for a run. To create a portable archive of the vine modulation study, run the following from the repository root.

```bash
python projects/vine_modulation/export_code.py . HEAD methods_code.zip
```

The archive contains the modulation estimators, the study project, the MIT license and a manifest. Each manifest entry identifies the repository path and SHA-256 digest of the file. The project README is also available at the root of the archive. Its worked examples and figure commands run after extraction.

For the numerical environments, source recordings and inclusion criteria, see the [study guide](user-guide/vine-modulation.md) and [research data guide](data-release.md).
