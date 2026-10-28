//! A mesh conversion (`mesh_convert`) can carry the surfaces picked on its mesh: `picks`, each a
//! `point` on the mesh in the body's coordinates and the `kind` of surface there (`auto`,
//! `plane`, `cylinder`, `cone` or `sphere`), and `all_surfaces`, which recognises every surface
//! first. A design can also hold a `mesh_fit` feature: the same `body`, `tolerance`, `picks` and
//! `all_surfaces`, making one solid per picked surface, or with `surfaces` one sheet per picked surface. Older designs need no transformation: without them the mesh converts as before. The
//! last release refuses a design saved with them (the newer format number).
