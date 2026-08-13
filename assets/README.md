# `assets/`

Tracked **data resources** that are not downloaded model weights (those live in
`models/`, which is fetched by `scripts/setup/` and gitignored) and not source.

One rule: a file belongs here when it is small, versioned, and consumed as data
rather than compiled in.

## `slanet_plus_dict.txt`

The 48-token table-structure vocabulary for **SLANet-plus**.

It is NOT the dictionary the table path currently uses. The active SLANeXt
vocabulary is compiled into `src/analysis/table/slanext/slanext_dict.cpp`
(`kDefaultDictText`), and the two genuinely differ — this file carries
` colspan="20"` and `<td></td>`, the compiled one carries `<td>` and
` colspan="25"`. They are different models' vocabularies, not two copies of one,
so neither can be derived from the other.

Kept because it has no other copy in the tree and a SLANet-plus checkpoint needs
it to decode. It has no consumer in the build today: point a deployment at it
explicitly if you run that model.
