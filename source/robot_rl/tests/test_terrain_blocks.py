"""Unit tests for the composable terrain-blocks system.

These tests exercise the block geometry, the composite assembler, and the
:class:`MetaCompositeTerrainImporter`'s per-block AABB/stair-span tensors and
``skill_probs_at`` / ``_project_world`` queries. Isaac sim is not required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    BlockChoice,
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    RandomizedCompositeSubTerrainCfg,
    SlopeBlockCfg,
    StairBlockCfg,
    composite_terrain,
    randomized_composite_terrain,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_composite_importer import (
    MetaCompositeTerrainImporter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_cfg(size_xy: tuple[float, float]):
    """Build a minimal stand-in for the importer's ``self.cfg`` attribute."""
    class _FakeGen:
        size = size_xy

    class _FakeImporterCfg:
        terrain_generator = _FakeGen()

    return _FakeImporterCfg()


def _attach_importer(
    cells: dict[tuple[int, int], dict],
    grid_shape: tuple[int, int],
    sub_size: tuple[float, float],
    skill_list: list[str],
) -> MetaCompositeTerrainImporter:
    """Build a :class:`MetaCompositeTerrainImporter` from raw cell metadata."""
    m = object.__new__(MetaCompositeTerrainImporter)
    m.device = "cpu"
    m.skill_list = list(skill_list)
    m.terrain_origins = torch.zeros((*grid_shape, 3))
    m.terrain_meta_data = dict(cells)
    m.cfg = _make_fake_cfg(sub_size)
    m._post_process_meta_data()
    return m


def _stair_up_cfg(size_x: float = 4.0) -> StairBlockCfg:
    """Return a deterministic stair-up cfg for tests.

    Uses ``step_depth = 0.5`` so the default ``size_x = 4.0`` divides
    exactly into 8 treads (the new composite ``StairBlock`` requires
    ``size_x`` to be an integer multiple of ``step_depth``).
    """
    return StairBlockCfg(
        size_x=size_x,
        direction="up",
        skill_probs={"stair_up": 1.0},
        step_dim_options=[(0.2, 0.5)],
        stair_width_range=(1.5, 1.5),
    )


def _flat_cfg(size_x: float = 3.0, skill_probs: dict | None = None) -> FlatBlockCfg:
    """Return a flat-block cfg."""
    sp = skill_probs or {"walk_forward": 1.0}
    return FlatBlockCfg(size_x=size_x, skill_probs=sp)


# ---------------------------------------------------------------------------
# Block geometry
# ---------------------------------------------------------------------------


def test_flat_block_aabb_and_origin():
    cfg = _flat_cfg(size_x=2.5)
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(1.0, 0.0), subterrain_size_y=3.0)
    assert out.aabb == (1.0, 3.5, 0.0, 3.0)
    assert out.needs_projection is False
    assert out.needs_directional_cmd is False
    # Default flat block (no flat_width_range) spans full subterrain y.
    assert out.extras == {"flat_width": 3.0}
    np.testing.assert_allclose(out.origin, [2.25, 1.5, 0.0])
    assert len(out.meshes) == 1


def test_stair_block_extras_and_aabb():
    np.random.seed(0)
    cfg = _stair_up_cfg(size_x=4.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(2.0, 0.0), subterrain_size_y=2.0)

    assert out.aabb == (2.0, 6.0, 0.0, 2.0)
    assert out.needs_projection is True
    assert out.needs_directional_cmd is True

    extras = out.extras
    assert extras["is_stair"] is True
    assert extras["direction"] == "up"
    centers = extras["stair_top_centers"]
    assert centers.shape[1] == 3
    # All centers shifted by x_offset=2.0.
    assert torch.all(centers[:, 0] >= 2.0)
    # Step depth divides the block exactly.
    step_depth = float(extras["stair_dimension"][1])
    assert pytest.approx(step_depth * extras["num_steps"], rel=1e-6) == 4.0


# ---------------------------------------------------------------------------
# Composite assembler
# ---------------------------------------------------------------------------


def test_composite_rejects_bad_size_sum():
    cfg = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_flat_cfg(4.0), _flat_cfg(4.0)],
    )
    with pytest.raises(ValueError, match="does not match"):
        composite_terrain(0.0, cfg)


def test_composite_rejects_empty_blocks():
    cfg = CompositeSubTerrainCfg(size=(10.0, 2.0), proportion=1.0, blocks=[])
    with pytest.raises(ValueError, match="at least one block"):
        composite_terrain(0.0, cfg)


def test_composite_rejects_bad_skill_probs():
    cfg = CompositeSubTerrainCfg(
        size=(2.0, 2.0),
        proportion=1.0,
        blocks=[_flat_cfg(2.0, skill_probs={"walk_forward": 0.5})],  # sums to 0.5
    )
    with pytest.raises(ValueError, match="sums to"):
        composite_terrain(0.0, cfg)


def test_composite_three_blocks():
    np.random.seed(1)
    cfg = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0), _stair_up_cfg(4.0)],
    )
    meshes, origin, md = composite_terrain(0.5, cfg)
    assert len(md["blocks"]) == 3
    assert md["blocks"][0]["aabb"] == (0.0, 4.0, 0.0, 2.0)
    assert md["blocks"][1]["aabb"] == (4.0, 6.0, 0.0, 2.0)
    assert md["blocks"][2]["aabb"] == (6.0, 10.0, 0.0, 2.0)
    assert md["blocks"][0]["needs_projection"] is True
    assert md["blocks"][1]["needs_projection"] is False
    assert md["blocks"][2]["needs_projection"] is True
    assert origin is not None
    assert len(meshes) > 1


# ---------------------------------------------------------------------------
# Z chaining across blocks
# ---------------------------------------------------------------------------


def _stair_down_cfg(size_x: float = 4.0) -> StairBlockCfg:
    """Stair-down variant for chaining tests."""
    cfg = _stair_up_cfg(size_x)
    cfg.direction = "down"
    cfg.skill_probs = {"stair_down": 1.0}
    return cfg


def test_flat_block_honors_base_z():
    """A flat block's mesh and metadata sit at the requested base_z."""
    np.random.seed(0)
    cfg = _flat_cfg(size_x=2.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=1.5,
    )
    assert out.entry_z == pytest.approx(1.5)
    assert out.exit_z == pytest.approx(1.5)
    assert out.origin[2] == pytest.approx(1.5)
    z_min, z_max = out.meshes[0].bounds[:, 2]
    assert z_min == pytest.approx(1.5, abs=1e-5)
    assert z_max == pytest.approx(1.5, abs=1e-5)


def test_stair_up_honors_base_z():
    """Stair-up: leftmost tread sits one step above ``base_z``; ``entry_z =
    base_z`` (the connecting-ground level); ``exit_z = base_z + N * step_h``.
    """
    np.random.seed(0)
    cfg = _stair_up_cfg(size_x=4.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.5,
    )
    centers = out.extras["stair_top_centers"]
    step_h, _, _ = out.extras["stair_dimension"]
    num_steps = out.extras["num_steps"]
    assert float(centers[0, 2]) == pytest.approx(0.5 + step_h)
    assert float(centers[-1, 2]) == pytest.approx(0.5 + num_steps * step_h)
    assert out.entry_z == pytest.approx(0.5)
    assert out.exit_z == pytest.approx(0.5 + num_steps * step_h)


def test_stair_down_honors_base_z():
    """Stair-down: leftmost (highest) tread sits one step below ``base_z``;
    ``entry_z = base_z``; ``exit_z = base_z - N * step_h``.
    """
    np.random.seed(0)
    cfg = _stair_down_cfg(size_x=4.0)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=2.0,
    )
    centers = out.extras["stair_top_centers"]
    step_h, _, _ = out.extras["stair_dimension"]
    num_steps = out.extras["num_steps"]
    assert float(centers[0, 2]) == pytest.approx(2.0 - step_h)
    assert float(centers[-1, 2]) == pytest.approx(2.0 - num_steps * step_h)
    assert out.entry_z == pytest.approx(2.0)
    assert out.exit_z == pytest.approx(2.0 - num_steps * step_h)


def test_composite_threads_z_across_blocks():
    """stair-up → flat → stair-down forms a continuous elevation profile."""
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0), _stair_down_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, cfg)
    blocks = md["blocks"]
    assert blocks[0]["entry_z"] == pytest.approx(0.0)
    assert blocks[0]["exit_z"] > 0.0
    # Flat platform inherits the stair-up's exit height (chain in action).
    assert blocks[1]["entry_z"] == pytest.approx(blocks[0]["exit_z"])
    assert blocks[1]["exit_z"] == pytest.approx(blocks[0]["exit_z"])
    # Stair-down enters at the flat's height and descends back to 0.
    assert blocks[2]["entry_z"] == pytest.approx(blocks[1]["exit_z"])
    assert blocks[2]["exit_z"] == pytest.approx(0.0, abs=1e-5)


def test_composite_start_z_offsets_first_block():
    """``start_z`` shifts the entire chain so the first block starts elevated."""
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(4.0, 2.0),
        proportion=1.0,
        start_z=3.0,
        blocks=[_flat_cfg(4.0)],
    )
    _, origin, md = composite_terrain(0.5, cfg)
    assert md["blocks"][0]["entry_z"] == pytest.approx(3.0)
    assert md["blocks"][0]["exit_z"] == pytest.approx(3.0)
    assert origin[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Stair-block platforms
# ---------------------------------------------------------------------------


def _stair_up_with_platforms_cfg(
    size_x: float, start_plat: float, end_plat: float
) -> StairBlockCfg:
    cfg = _stair_up_cfg(size_x)
    cfg.start_platform_length = start_plat
    cfg.end_platform_length = end_plat
    return cfg


def test_stair_block_rejects_oversize_platforms():
    cfg = _stair_up_with_platforms_cfg(size_x=2.0, start_plat=1.0, end_plat=1.5)
    with pytest.raises(ValueError, match="no positive x extent"):
        cfg.class_type(cfg, difficulty=0.0).build(
            local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.0,
        )


def test_stair_block_platforms_shrink_stair_region():
    """Stairs occupy size_x - start_platform - end_platform; extras report boundaries."""
    np.random.seed(0)
    cfg = _stair_up_with_platforms_cfg(size_x=6.0, start_plat=1.0, end_plat=0.5)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.0,
    )
    # AABB still covers the full block.
    assert out.aabb == (0.0, 6.0, 0.0, 2.0)
    # Stair tread region boundaries.
    assert out.extras["start_platform_xmax"] == pytest.approx(1.0)
    assert out.extras["end_platform_xmin"] == pytest.approx(5.5)
    # Stairs filled the inner 4.5m, so the first tread should sit just past
    # the start platform.
    centers = out.extras["stair_top_centers"]
    step_d = out.extras["stair_dimension"][1]
    assert float(centers[0, 0]) >= 1.0
    assert float(centers[0, 0]) <= 1.0 + step_d + 1e-5


def test_stair_block_platforms_sit_at_entry_and_exit_z():
    """Start platform at entry_z (= base_z); end platform at exit_z."""
    np.random.seed(0)
    cfg = _stair_up_with_platforms_cfg(size_x=6.0, start_plat=1.0, end_plat=0.5)
    out = cfg.class_type(cfg, difficulty=0.0).build(
        local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0, base_z=0.3,
    )
    # First inserted mesh is the start platform; last is the end platform.
    start_plat_mesh = out.meshes[0]
    end_plat_mesh = out.meshes[-1]
    # Each platform plate is essentially zero-thickness in z.
    sp_z = float(start_plat_mesh.bounds[0, 2])
    ep_z = float(end_plat_mesh.bounds[0, 2])
    assert sp_z == pytest.approx(out.entry_z, abs=1e-5)
    assert ep_z == pytest.approx(out.exit_z, abs=1e-5)


def test_project_world_platform_heights():
    """Start platform (at entry_z) keeps the foot's x; end platform (at
    exit_z) resolves to the top tread by height."""
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(6.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_with_platforms_cfg(size_x=6.0, start_plat=1.0, end_plat=0.5)],
    )
    _, _, md = composite_terrain(0.5, cfg)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(6.0, 2.0),
        skill_list=["stair_up"],
    )

    # Cell (0, 0): local_x = world_x + 3 (offset_x = -3).
    centers = md["blocks"][0]["extras"]["stair_top_centers"]
    entry_z = float(md["blocks"][0]["entry_z"])
    exit_z = float(centers[-1, 2])
    last_tread_x = float(centers[-1, 0])

    # Start platform (local_x=0.5 → world -2.5) at the entry height: it's
    # in-span, so the foot's x is kept and z = entry_z.
    out = m._project_world(torch.tensor([[-2.5, 0.0, entry_z]]))
    assert out[0, 0].item() == pytest.approx(-2.5)
    assert out[0, 2].item() == pytest.approx(entry_z, abs=1e-5)

    # End platform (local_x=5.7 → world 2.7) at the exit height: height
    # resolves to the top tread → its center x and exit_z.
    out = m._project_world(torch.tensor([[2.7, 0.0, exit_z]]))
    assert out[0, 2].item() == pytest.approx(exit_z, abs=1e-5)
    assert out[0, 0].item() == pytest.approx(last_tread_x - 3.0)


def test_project_world_tread_by_height():
    """A foot whose xy and height both sit on a tread snaps to that tread top."""
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(4.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, cfg)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(4.0, 2.0),
        skill_list=["stair_up"],
    )
    centers = md["blocks"][0]["extras"]["stair_top_centers"]
    t_x, t_z = float(centers[3, 0]), float(centers[3, 2])   # tread 3
    out = m._project_world(torch.tensor([[t_x - 2.0, 0.0, t_z]]))   # offset_x = -2
    assert out[0, 2].item() == pytest.approx(t_z, abs=1e-5)
    assert out[0, 0].item() == pytest.approx(t_x - 2.0)


def test_importer_stores_per_block_z():
    """The importer captures entry_z / exit_z so the debug viz can use them."""
    np.random.seed(0)
    sub = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0), _stair_down_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(10.0, 2.0),
        skill_list=["stair_up", "walk_forward", "stair_down"],
    )
    assert m._block_exit_z[0, 0, 0].item() > 0.0
    assert m._block_entry_z[0, 0, 1].item() == pytest.approx(
        m._block_exit_z[0, 0, 0].item()
    )
    assert m._block_exit_z[0, 0, 1].item() == pytest.approx(
        m._block_entry_z[0, 0, 1].item()
    )
    assert m._block_exit_z[0, 0, 2].item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MetaCompositeTerrainImporter
# ---------------------------------------------------------------------------


def test_importer_builds_block_tensors_from_composite():
    np.random.seed(2)
    sub = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0, {"walk_forward": 1.0}), _stair_up_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, sub)

    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(10.0, 2.0),
        skill_list=["stair_up", "walk_forward"],
    )

    # 3 blocks, full footprint.
    assert m._block_valid[0, 0].tolist() == [True, True, True]
    aabbs = m._block_aabbs[0, 0].tolist()
    assert aabbs[0][:2] == [0.0, 4.0]
    assert aabbs[1][:2] == [4.0, 6.0]
    assert aabbs[2][:2] == [6.0, 10.0]

    # Two stair spans, one flat in between.
    assert m._stair_span_valid[0, 0].tolist() == [True, True]
    spans = m._stair_span_xrange[0, 0].tolist()
    assert spans[0] == [0.0, 4.0]
    assert spans[1] == [6.0, 10.0]


def test_importer_handles_legacy_schema():
    """A cell using the legacy schema (no ``blocks`` key) should still work."""
    legacy_md = {
        "needs_projection": False,
        "needs_directional_cmd": False,
        "is_border": False,
        "skill_probs": {"walk_forward": 1.0},
    }
    m = _attach_importer(
        {(0, 0): legacy_md}, grid_shape=(1, 1), sub_size=(10.0, 2.0),
        skill_list=["walk_forward"],
    )
    # One synthesized block covering the full cell.
    assert m._block_valid[0, 0].tolist()[0] is True
    assert m._block_aabbs[0, 0, 0].tolist() == [0.0, 10.0, 0.0, 2.0]
    assert m._block_skill_probs[0, 0, 0, 0].item() == 1.0
    # No stair spans.
    assert not m._stair_span_valid[0, 0].any().item()


def test_skill_probs_at_resolves_block_by_xy():
    """``skill_probs_at`` returns the right block's distribution."""
    np.random.seed(3)
    sub = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0, {"walk_forward": 1.0}), _stair_up_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(10.0, 2.0),
        skill_list=["stair_up", "walk_forward"],
    )

    # World-frame grid is centered at origin: cell (0,0) spans [-5, 5] x [-1, 1].
    # local_x = world_x + 5 (cell xmin in world coords is -5).
    queries = torch.tensor([
        [-3.0, 0.0],  # local_x=2  -> stair_up block 0
        [ 0.0, 0.0],  # local_x=5  -> flat block 1
        [ 3.0, 0.0],  # local_x=8  -> stair_up block 2
    ])
    probs = m.skill_probs_at(queries)
    expected = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    torch.testing.assert_close(probs, expected)


def test_project_world_span_resolution():
    """xy selects the span; a foot on a tread of each span (xy+z consistent)
    resolves to that span's tread."""
    np.random.seed(4)
    sub = CompositeSubTerrainCfg(
        size=(10.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(2.0, {"walk_forward": 1.0}), _stair_up_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(10.0, 2.0),
        skill_list=["stair_up", "walk_forward"],
    )
    c0 = md["blocks"][0]["extras"]["stair_top_centers"]   # span 0, local x [0, 4)
    c2 = md["blocks"][2]["extras"]["stair_top_centers"]   # span 1, local x [6, 10)

    # Foot on span 0, tread 3.
    out = m._project_world(torch.tensor([[float(c0[3, 0]) - 5.0, 0.0, float(c0[3, 2])]]))
    assert out[0, 0].item() == pytest.approx(float(c0[3, 0]) - 5.0)
    assert out[0, 2].item() == pytest.approx(float(c0[3, 2]), abs=1e-5)

    # Foot on span 1, tread 3 (its z is threaded up from span 0).
    out = m._project_world(torch.tensor([[float(c2[3, 0]) - 5.0, 0.0, float(c2[3, 2])]]))
    assert out[0, 0].item() == pytest.approx(float(c2[3, 0]) - 5.0)
    assert out[0, 2].item() == pytest.approx(float(c2[3, 2]), abs=1e-5)


def test_project_world_z_exit_fictitious():
    """A foot past the top of the stairs (xy beyond span_xmax) at the
    top-of-stairs height → exit-side fictitious tread at exit_z."""
    np.random.seed(0)
    sub = CompositeSubTerrainCfg(
        size=(7.0, 2.0),
        proportion=1.0,
        blocks=[_stair_up_cfg(4.0), _flat_cfg(3.0)],
    )
    _, _, md = composite_terrain(0.5, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(7.0, 2.0),
        skill_list=["walk_forward", "stair_up"],
    )
    # stair span [0, 4), flat [4, 7).  Cell offset_x = -3.5.
    step_depth = float(md["blocks"][0]["extras"]["stair_dimension"][1])
    exit_z = float(md["blocks"][0]["extras"]["stair_top_centers"][-1, 2])

    # Foot on the flat past the stair (local_x=5.0 ≥ span_xmax=4) at exit_z.
    out = m._project_world(torch.tensor([[5.0 - 3.5, 0.0, exit_z]]))
    assert out[0, 0].item() == pytest.approx((4.0 + 0.5 * step_depth) - 3.5)
    assert out[0, 2].item() == pytest.approx(exit_z, abs=1e-5)


def test_project_world_passthrough_when_no_stair_in_cell():
    """Cells with zero stair spans pass through ``(x, y, 0)`` — the query
    z does not pull the result onto any (non-existent) stair."""
    np.random.seed(0)
    sub = CompositeSubTerrainCfg(
        size=(4.0, 2.0),
        proportion=1.0,
        blocks=[_flat_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.0, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(4.0, 2.0),
        skill_list=["walk_forward"],
    )

    queries = torch.tensor([[-1.5, 0.5, 0.3], [0.0, -0.3, 1.0], [1.7, 0.0, 0.0]])
    out = m._project_world(queries)
    torch.testing.assert_close(out[:, 2], torch.zeros(3))
    # x and y pass through (z column is dropped to 0).
    torch.testing.assert_close(out[:, :2], queries[:, :2])


def test_project_world_nearest_span_selection():
    """In a cell with two stair spans, a between-spans point resolves
    against whichever span is nearer by ``|local_x − span_center|``; height
    then picks the level on that span."""
    np.random.seed(4)
    sub = CompositeSubTerrainCfg(
        size=(11.0, 2.0),
        proportion=1.0,
        blocks=[
            _stair_up_cfg(2.0),                          # span [0, 2),  center 1.0
            _flat_cfg(5.0, {"walk_forward": 1.0}),       # flat  [2, 7)
            _stair_up_cfg(4.0),                          # span [7, 11), center 9.0
        ],
    )
    _, _, md = composite_terrain(0.5, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(11.0, 2.0),
        skill_list=["stair_up", "walk_forward"],
    )

    # Cell offset_x = -5.5.  local_x=3.0 (dist 2 from span 0, 6 from span 1)
    # → span 0, past its xmax → exit-side fictitious at exit_z.
    sd0 = float(md["blocks"][0]["extras"]["stair_dimension"][1])
    exit_z0 = float(md["blocks"][0]["extras"]["stair_top_centers"][-1, 2])
    out = m._project_world(torch.tensor([[3.0 - 5.5, 0.0, exit_z0]]))
    assert out[0, 0].item() == pytest.approx((2.0 + 0.5 * sd0) - 5.5)
    assert out[0, 2].item() == pytest.approx(exit_z0, abs=1e-5)

    # local_x=6.0 (dist 5 from span 0, 3 from span 1) → span 1, before its
    # xmin, at entry height → entry-side fictitious at entry_z.
    sd1 = float(md["blocks"][2]["extras"]["stair_dimension"][1])
    entry_z1 = float(md["blocks"][2]["entry_z"])
    out = m._project_world(torch.tensor([[6.0 - 5.5, 0.0, entry_z1]]))
    assert out[0, 0].item() == pytest.approx((7.0 - 0.5 * sd1) - 5.5)
    assert out[0, 2].item() == pytest.approx(entry_z1, abs=1e-5)


def _flat_then_stair_importer():
    """Build a ``[flat(3) | stair_up(4)]`` importer in a 7×2 cell.

    Cell origin is centred, so ``offset_x = -3.5`` (world_x = local_x − 3.5).
    Flat block spans local x [0, 3); stair span is [3, 7) with no start
    platform (riser at local x = 3.0).  Returns ``(importer, md)``.
    """
    np.random.seed(0)
    sub = CompositeSubTerrainCfg(
        size=(7.0, 2.0),
        proportion=1.0,
        blocks=[_flat_cfg(3.0), _stair_up_cfg(4.0)],
    )
    _, _, md = composite_terrain(0.5, sub)
    m = _attach_importer(
        {(0, 0): md}, grid_shape=(1, 1), sub_size=(7.0, 2.0),
        skill_list=["walk_forward", "stair_up"],
    )
    return m, md


def test_project_world_z_picks_tread_when_ankle_over_flat():
    """The key fix: ankle xy still over the flat (before the riser) but the
    foot is at the first tread's height → the xyz projection snaps to the
    real first tread, NOT the fictitious entry tread a full step below."""
    m, md = _flat_then_stair_importer()
    centers = md["blocks"][1]["extras"]["stair_top_centers"]
    tread0_x_local = float(centers[0, 0])   # 3.0 + step_depth/2
    tread0_z = float(centers[0, 2])         # base_z + step_height (> entry_z)

    # Ankle over the flat: local_x = 2.5 (world -1.0), but foot height is
    # the first tread top.  xy-only would give entry_z; xyz must give tread 0.
    out = m._project_world(torch.tensor([[2.5 - 3.5, 0.0, tread0_z]]))
    assert out[0, 2].item() == pytest.approx(tread0_z, abs=1e-5)
    assert out[0, 0].item() == pytest.approx(tread0_x_local - 3.5)

    # The whole-step error this fixes: the fictitious entry level sits one
    # step below the foot, so a height-blind projection would have placed
    # the reference there.
    entry_z = float(md["blocks"][1]["entry_z"])
    assert entry_z < tread0_z


def test_project_world_z_on_flat_picks_fictitious_entry():
    """Ankle over the flat AND foot at the connecting-ground height →
    fictitious entry tread (height nearest entry_z)."""
    m, md = _flat_then_stair_importer()
    entry_z = float(md["blocks"][1]["entry_z"])
    step_depth = float(md["blocks"][1]["extras"]["stair_dimension"][1])

    out = m._project_world(torch.tensor([[2.5 - 3.5, 0.0, entry_z]]))
    assert out[0, 2].item() == pytest.approx(entry_z, abs=1e-5)
    # Fictitious entry x sits half a step-depth before the riser.
    assert out[0, 0].item() == pytest.approx((3.0 - 0.5 * step_depth) - 3.5)


def test_project_world_z_matches_full_tread():
    """A foot fully on a tread (xy and height agree) returns that tread."""
    m, md = _flat_then_stair_importer()
    centers = md["blocks"][1]["extras"]["stair_top_centers"]
    t2_x_local = float(centers[2, 0])
    t2_z = float(centers[2, 2])

    out = m._project_world(torch.tensor([[t2_x_local - 3.5, 0.0, t2_z]]))
    assert out[0, 0].item() == pytest.approx(t2_x_local - 3.5, abs=1e-5)
    assert out[0, 2].item() == pytest.approx(t2_z, abs=1e-5)


def test_project_world_single_point_signature():
    """Non-stair cells pass through (x, y, 0) for an (N, 3) query."""
    legacy_md = {"is_border": False, "skill_probs": {"walk_forward": 1.0}}
    m = _attach_importer(
        {(0, 0): legacy_md}, grid_shape=(1, 1), sub_size=(4.0, 2.0),
        skill_list=["walk_forward"],
    )
    pts = torch.tensor([[0.5, 0.1, 0.0], [-1.0, 0.0, 0.4]])
    out = m._project_world(pts)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out[:, 2], torch.zeros(2))


def test_project_world_rejects_wrong_shape():
    legacy_md = {"is_border": False, "skill_probs": {"walk_forward": 1.0}}
    m = _attach_importer(
        {(0, 0): legacy_md}, grid_shape=(1, 1), sub_size=(4.0, 2.0),
        skill_list=["walk_forward"],
    )
    with pytest.raises(ValueError, match="shape"):
        m._project_world(torch.zeros((1, 2)))  # missing z column
    with pytest.raises(ValueError, match="shape"):
        m._project_world(torch.zeros((1, 1, 3)))  # extra axis


# ---------------------------------------------------------------------------
# Randomized composite sub-terrain
# ---------------------------------------------------------------------------


def _flat_choice_cfg(weight: float = 1.0) -> BlockChoice:
    return BlockChoice(
        cfg=FlatBlockCfg(skill_probs={"walk_forward": 1.0}),
        weight=weight,
    )


def _stair_choice_cfg(weight: float = 1.0) -> BlockChoice:
    return BlockChoice(
        cfg=StairBlockCfg(
            direction="up",
            skill_probs={"stair_up": 1.0},
            step_dim_options=[(0.135, 0.233)],
            stair_width_range=(1.5, 1.5),
        ),
        weight=weight,
    )


def _slope_choice_cfg(weight: float = 1.0) -> BlockChoice:
    return BlockChoice(
        cfg=SlopeBlockCfg(
            direction="up",
            skill_probs={"walk_forward": 1.0},
            rise_range=(0.3, 0.5),
            slope_width_range=(1.5, 1.5),
        ),
        weight=weight,
    )


def _randomized_cfg(size_x: float = 12.0, **overrides) -> RandomizedCompositeSubTerrainCfg:
    base = dict(
        proportion=1.0,
        size=(size_x, 2.0),
        origin_block_index=0,
        force_flat_origin=True,
        length_range=(1.0, 2.5),
        choices=[_flat_choice_cfg(2.0), _stair_choice_cfg(1.0), _slope_choice_cfg(1.0)],
    )
    base.update(overrides)
    return RandomizedCompositeSubTerrainCfg(**base)


def test_randomized_composite_length_sum():
    np.random.seed(0)
    cfg = _randomized_cfg(size_x=12.0)
    meshes, origin, meta = randomized_composite_terrain(0.0, cfg)
    total = sum(b.size_x for b in cfg.blocks)
    assert pytest.approx(total, abs=1e-5) == 12.0
    assert len(meshes) > 0
    assert origin is not None
    assert len(meta["blocks"]) == len(cfg.blocks)


def test_randomized_composite_deterministic():
    np.random.seed(42)
    cfg_a = _randomized_cfg(size_x=10.0)
    randomized_composite_terrain(0.0, cfg_a)
    types_a = [type(b).__name__ for b in cfg_a.blocks]
    lengths_a = [b.size_x for b in cfg_a.blocks]

    np.random.seed(42)
    cfg_b = _randomized_cfg(size_x=10.0)
    randomized_composite_terrain(0.0, cfg_b)
    types_b = [type(b).__name__ for b in cfg_b.blocks]
    lengths_b = [b.size_x for b in cfg_b.blocks]

    assert types_a == types_b
    np.testing.assert_allclose(lengths_a, lengths_b, atol=1e-6)


def test_randomized_composite_zero_weights_raises():
    np.random.seed(0)
    cfg = _randomized_cfg(
        force_flat_origin=False,
        choices=[_flat_choice_cfg(0.0), _stair_choice_cfg(0.0)],
    )
    with pytest.raises(ValueError, match="weights are zero"):
        randomized_composite_terrain(0.0, cfg)


def test_randomized_composite_empty_choices_raises():
    np.random.seed(0)
    cfg = _randomized_cfg(choices=[])
    with pytest.raises(ValueError, match="non-empty"):
        randomized_composite_terrain(0.0, cfg)


def test_randomized_composite_per_choice_length_override():
    np.random.seed(7)
    # Flat block with a tight length-range override — exercises the
    # per-choice length_range path independently of stair rounding (which
    # always rounds DOWN to N*step_depth and so can't be used to assert a
    # tight upper bound on sampled length).
    tight_flat = BlockChoice(
        cfg=FlatBlockCfg(skill_probs={"walk_forward": 1.0}),
        weight=1.0,
        length_range=(0.8, 0.81),
    )
    cfg = _randomized_cfg(
        size_x=10.0,
        force_flat_origin=False,
        choices=[tight_flat],
        length_range=(5.0, 5.0),  # global range, ignored by tight_flat
    )
    randomized_composite_terrain(0.0, cfg)
    for b in cfg.blocks[:-1]:
        # Every non-tail block must respect the tight override (tail may
        # absorb the remainder via the stretch policy).
        assert 0.8 - 1e-6 <= b.size_x <= 0.81 + 1e-6


def test_randomized_composite_force_flat_origin():
    # The forced spawn block must inherit the configured FlatBlockCfg from
    # ``choices`` (skill_probs, flat_width_range, etc.) — not the legacy
    # ``FlatBlockCfg()`` default, which would silently re-introduce
    # running-eligible flat skill_probs on the spawn block.
    custom_skill_probs = {"walk_forward": 0.9, "standing": 0.1}
    for seed in (1, 2, 3, 4, 5):
        np.random.seed(seed)
        cfg = _randomized_cfg(
            size_x=8.0,
            force_flat_origin=True,
            choices=[
                BlockChoice(
                    cfg=FlatBlockCfg(skill_probs=dict(custom_skill_probs)),
                    weight=1.0,
                ),
                _stair_choice_cfg(1.0),
            ],
            length_range=(1.0, 2.0),
        )
        randomized_composite_terrain(0.0, cfg)
        assert isinstance(cfg.blocks[0], FlatBlockCfg)
        assert cfg.blocks[0].skill_probs == custom_skill_probs


def test_randomized_composite_force_flat_origin_without_flat_choice_raises():
    # With ``force_flat_origin=True`` and no FlatBlockCfg in ``choices`` the
    # builder used to silently fall back to a default FlatBlockCfg (which
    # advertises running). Now it surfaces the inconsistency.
    np.random.seed(0)
    cfg = _randomized_cfg(
        size_x=8.0,
        force_flat_origin=True,
        choices=[_stair_choice_cfg(1.0)],
        length_range=(1.0, 2.0),
    )
    with pytest.raises(ValueError, match="FlatBlockCfg"):
        randomized_composite_terrain(0.0, cfg)


def test_randomized_composite_distribution_sanity():
    np.random.seed(123)
    type_counts = {"FlatBlockCfg": 0, "StairBlockCfg": 0}
    # Use disjoint per-choice length so we tile evenly.
    choices = [
        BlockChoice(
            cfg=FlatBlockCfg(skill_probs={"walk_forward": 1.0}),
            weight=3.0,
        ),
        BlockChoice(
            cfg=StairBlockCfg(
                direction="up",
                skill_probs={"stair_up": 1.0},
                step_dim_options=[(0.135, 0.233)],
                stair_width_range=(1.5, 1.5),
            ),
            weight=1.0,
        ),
    ]
    for _ in range(200):
        cfg = RandomizedCompositeSubTerrainCfg(
            proportion=1.0,
            size=(20.0, 2.0),
            origin_block_index=0,
            force_flat_origin=False,
            length_range=(1.0, 1.0001),
            choices=choices,
        )
        randomized_composite_terrain(0.0, cfg)
        for b in cfg.blocks:
            type_counts[type(b).__name__] += 1
    total = sum(type_counts.values())
    flat_frac = type_counts["FlatBlockCfg"] / total
    # Expected 0.75, allow ±0.05 with 200 * ~20 = 4000 samples.
    assert abs(flat_frac - 0.75) < 0.05


# ---------------------------------------------------------------------------
# Stair partial cut-off
# ---------------------------------------------------------------------------


def test_stair_block_exact_tiling_no_partial():
    """Composite ``StairBlock`` requires ``size_x = N * step_depth`` exactly
    and never emits a partial trailing tread. Each tread is full depth.
    """
    np.random.seed(0)
    cfg = StairBlockCfg(
        size_x=3.0,  # exactly 3 treads of step_depth 1.0
        direction="up",
        skill_probs={"stair_up": 1.0},
        step_dim_options=[(0.15, 1.0)],
        stair_width_range=(1.5, 1.5),
    )
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0)
    extras = out.extras
    assert extras["num_steps"] == 3
    step_depth = float(extras["stair_dimension"][1])
    assert pytest.approx(step_depth, abs=1e-6) == 1.0
    # No partial tread ever.
    assert extras["partial_step_length"] is None
    # entry_z is the connecting ground; exit_z is N * step_h above it.
    assert pytest.approx(out.entry_z, abs=1e-6) == 0.0
    assert pytest.approx(out.exit_z, abs=1e-6) == 3 * 0.15


def test_stair_block_size_x_not_multiple_raises():
    """If ``size_x`` is not an integer multiple of ``step_depth``, the
    composite StairBlock raises rather than silently rescaling.
    """
    np.random.seed(0)
    cfg = StairBlockCfg(
        size_x=3.1,
        direction="up",
        skill_probs={"stair_up": 1.0},
        step_dim_options=[(0.15, 1.0)],
        stair_width_range=(1.5, 1.5),
    )
    with pytest.raises(ValueError, match="integer multiple of step_depth"):
        cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0)


def test_stair_block_chain_continuous_staircase():
    """Two back-to-back stair blocks chain into a single continuous staircase:
    block B's first tread sits one step above block A's last tread.
    """
    np.random.seed(0)
    cfg = CompositeSubTerrainCfg(
        size=(6.0, 2.0),
        proportion=1.0,
        blocks=[
            StairBlockCfg(
                size_x=3.0,  # 3 treads
                direction="up",
                skill_probs={"stair_up": 1.0},
                step_dim_options=[(0.15, 1.0)],
                stair_width_range=(1.5, 1.5),
            ),
            StairBlockCfg(
                size_x=3.0,  # 3 more treads
                direction="up",
                skill_probs={"stair_up": 1.0},
                step_dim_options=[(0.15, 1.0)],
                stair_width_range=(1.5, 1.5),
            ),
        ],
    )
    _, _, md = composite_terrain(0.0, cfg)
    a = md["blocks"][0]
    b = md["blocks"][1]
    # Block A's exit_z = base_z + N_A * step_h; block B's first tread top =
    # A.exit_z + step_h (the chain "continues" without any flat plateau).
    a_centers = a["extras"]["stair_top_centers"]
    b_centers = b["extras"]["stair_top_centers"]
    step_h = a["extras"]["stair_dimension"][0]
    assert pytest.approx(b["entry_z"], abs=1e-6) == a["exit_z"]
    assert pytest.approx(float(a_centers[-1, 2]), abs=1e-6) == a["exit_z"]
    assert pytest.approx(float(b_centers[0, 2]), abs=1e-6) == a["exit_z"] + step_h


# ---------------------------------------------------------------------------
# Slope block
# ---------------------------------------------------------------------------


def test_slope_block_aabb_and_z_thread():
    np.random.seed(0)
    cfg = SlopeBlockCfg(
        size_x=2.0,
        direction="up",
        skill_probs={"walk_forward": 1.0},
        rise_range=(0.5, 0.5),
        slope_width_range=(1.5, 1.5),
    )
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(1.0, 0.0), subterrain_size_y=2.0)
    assert out.aabb == (1.0, 3.0, 0.0, 2.0)
    assert out.needs_projection is False
    assert out.needs_directional_cmd is True
    assert pytest.approx(out.exit_z - out.entry_z, abs=1e-6) == 0.5
    assert out.extras["is_slope"] is True
    assert out.extras["direction"] == "up"


def test_slope_block_with_platforms():
    np.random.seed(0)
    cfg = SlopeBlockCfg(
        size_x=4.0,
        direction="down",
        skill_probs={"walk_forward": 1.0},
        rise_range=(0.4, 0.4),
        slope_width_range=(1.5, 1.5),
        start_platform_length=0.5,
        end_platform_length=0.5,
    )
    out = cfg.class_type(cfg, difficulty=0.0).build(local_origin_xy=(0.0, 0.0), subterrain_size_y=2.0)
    # 1 platform + 1 slope + 1 platform = 3 meshes.
    assert len(out.meshes) == 3
    # entry_z stays at base_z; exit_z descends.
    assert pytest.approx(out.entry_z, abs=1e-6) == 0.0
    assert pytest.approx(out.exit_z, abs=1e-6) == -0.4
    # Platforms shrink slope region.
    assert pytest.approx(out.extras["start_platform_xmax"], abs=1e-6) == 0.5
    assert pytest.approx(out.extras["end_platform_xmin"], abs=1e-6) == 3.5


def test_slope_block_inside_randomized_composite():
    np.random.seed(0)
    cfg = RandomizedCompositeSubTerrainCfg(
        proportion=1.0,
        size=(15.0, 2.0),
        origin_block_index=0,
        force_flat_origin=True,
        length_range=(2.0, 3.0),
        choices=[
            _flat_choice_cfg(1.0),
            _stair_choice_cfg(1.0),
            _slope_choice_cfg(1.0),
        ],
    )
    meshes, origin, meta = randomized_composite_terrain(0.0, cfg)
    assert any(b["block_type"] == "SlopeBlock" for b in meta["blocks"])
    # z-threading invariant: each block's entry_z equals previous exit_z.
    for prev, curr in zip(meta["blocks"], meta["blocks"][1:]):
        assert pytest.approx(curr["entry_z"], abs=1e-6) == prev["exit_z"]


