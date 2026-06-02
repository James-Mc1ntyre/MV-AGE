# Third-Party Notices

MV_AGE vendors small portions of third-party research code so the
exact method can run from a standalone checkout.

## Multiplex PageRank

- Path: `mv_age/vendor/Multiplex-PageRank/`
- Component: `multiplexPageRank.m`
- Copyright: Jacopo Iacovacci and Ginestra Bianconi
- License stated by upstream source: GNU General Public License, version 3
  or later
- Citation requested by upstream:
  A. Halu, R. J. Mondragon, P. Panzarasa, and G. Bianconi,
  "Multiplex pagerank." PLOS ONE 8, no. 10 (2013): e78293.

## MAGCN TensorFlow Implementation

- Path: `mv_age/vendor/MAGCN/`
- Upstream: https://github.com/sxu-yaokx/MAGCN
- Component: TensorFlow implementation of MAGCN
- Upstream description: author's implementation for "Multi-view graph
  convolutional networks with attention mechanism"
- Citation:
  Yao, K., Liang, J., Liang, J., Li, M., and Cao, F. (2022).
  "Multi-view graph convolutional networks with attention mechanism."
  Artificial Intelligence, 307, 103708.

The vendored MAGCN copy in this repository does not include an explicit
license file. Before distributing this package publicly, confirm the upstream
redistribution terms or replace this vendored implementation with code whose
license is documented.
