# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-off USD authoring tool: give every G1 rigid-body link with visual geometry a
convex-decomposition collider copied from its own visual mesh, and write the result to
``assets/g1_full_collision.usd`` (used by ``scripts/play_motion.py --dynamic``, see the
``G1_USD_PATH`` comment there).

The robot has the Unitree/Inspire 5-finger ("FTP" variant) hand instead of the 3-finger simplified
hand ``../GEM-X/third_party/soma-retargeter`` uses for its own IK target (that pipeline actually
retargets onto the hand-less ``g1_29dof_rev_1_0`` skeleton - a bare wrist with a static rubber-hand
mesh, no finger joints at all - so it never drives any hand model; see ``scripts/play_motion.py``'s
``load_motion()`` for how the missing finger columns are handled).

The source asset is ``assets/g1_inspire_hand_raw/g1_inspire_ftp.usd``, produced by converting
newton-assets' ``g1_29dof_rev_1_0_with_inspire_hand_FTP`` URDF with IsaacLab's URDF importer (this
step needs a running Isaac Sim/Kit process, unlike the rest of this script):

.. code-block:: bash

    # the URDF's mesh filenames are relative ("meshes/xxx.STL"), resolved relative to the URDF
    # file's own directory - copy it out of newton's "mjcf/" folder to sit next to "meshes/" first
    cp ~/.cache/newton/newton-assets_unitree_g1_308a72cd/unitree_g1/mjcf/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf \
       ~/.cache/newton/newton-assets_unitree_g1_308a72cd/unitree_g1/

    conda activate isaac
    python /home/glass/IsaacLab/scripts/tools/convert_urdf.py \
        ~/.cache/newton/newton-assets_unitree_g1_308a72cd/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf \
        assets/g1_inspire_hand_raw/g1_inspire_ftp.usd \
        --headless --merge-joints

This raw import has no collision geometry populated on most links (URDF importer leaves empty
``collisions`` placeholders under a ``Physics`` variant), so under physics-driven playback those
links pass straight through the ground/other bodies with no contact response. This script rebuilds
collision on every rigid-body link (54 total: 29 body DOF links + 2x12 finger links + torso/pelvis/
etc.) the same way, so the whole robot has consistent, close-fitting collision.

Note: a few links (e.g. the ``*_ankle_roll_link`` feet) already carry hand-placed collision
primitives (small spheres near the sole) imported straight from the URDF's own ``<collision>``
tags - deliberately minimal, and a common, numerically stable technique for legged-robot foot
contact. This script discards those and replaces them with a full convex/SDF mesh collider like
every other link, which gives fuller foot coverage but may make ground contact slightly less
stable (edge-catching, jitter) than the tuned original - watch foot behavior specifically if you
see new instability after regenerating.

Aside from the URDF-to-USD conversion above, this is plain USD authoring - no Isaac Sim/Kit app
needs to be running, just a Python with ``pxr`` (usd-core) importable, e.g. this project's own
``isaac`` conda env.

.. code-block:: bash

    python scripts/tools/add_g1_full_body_collision.py

Re-run this after ``assets/g1_inspire_hand_raw`` is regenerated (e.g. from a newer newton-assets
commit), since the output file is not committed to git (this repo's .gitignore excludes
``*.usd``).
"""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# The PhysxSDFMeshCollisionAPI Python schema class only exists inside a running Kit/Isaac Sim
# process (it's registered by the omni.usd.schema.physx extension at app startup), which this
# plain usd-core script does not launch. Applying it by name via AddAppliedSchema()/authoring the
# attribute directly writes the exact same USD data (the "apiSchemas" listOp token plus the
# namespaced attribute) that PhysxSchema.PhysxSDFMeshCollisionAPI.Apply() would produce inside Kit -
# Isaac Sim only cares about the on-disk USD contents, not which tool authored them.
_PHYSX_SDF_MESH_COLLISION_API = "PhysxSDFMeshCollisionAPI"

_REPO_ROOT = Path(__file__).parent.parent.parent
SRC = _REPO_ROOT / "assets" / "g1_inspire_hand_raw" / "g1_inspire_ftp.usd"
OUT = _REPO_ROOT / "assets" / "g1_full_collision.usd"
ROBOT_PRIM_PATH = "/g1_29dof_rev_1_0_with_inspire_hand_FTP"


def _copy_visual_mesh_to_link_space(link: Usd.Prim, visuals: Usd.Prim):
    """Collect every mesh under ``visuals``, transformed into ``link``'s local frame.

    Traversing with instance proxies is required: each link's visual mesh is referenced in as an
    instanceable prim, which a plain PrimRange()/authoring call would treat as opaque/read-only.
    Reading attributes (points, transforms) off an instance proxy is fine though, so we read the
    geometry here and re-author it as a fresh, non-instanced mesh under "collisions" afterwards.
    """
    world_to_link = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()

    points, counts, indices = [], [], []
    for prim in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        mesh_counts = mesh.GetFaceVertexCountsAttr().Get()
        mesh_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not mesh_points or not mesh_counts or not mesh_indices:
            continue
        mesh_to_link = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * world_to_link
        offset = len(points)
        points.extend(mesh_to_link.Transform(Gf.Vec3d(p)) for p in mesh_points)
        counts.extend(mesh_counts)
        indices.extend(int(i) + offset for i in mesh_indices)

    return points, counts, indices


def main():
    if not SRC.exists():
        raise FileNotFoundError(
            f"{SRC} not found. Generate it first by converting the Inspire-hand URDF to USD - see "
            "this module's docstring for the exact convert_urdf.py command."
        )

    stage = Usd.Stage.Open(str(SRC))
    robot = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not robot.IsValid():
        raise RuntimeError(f"robot prim not found at {ROBOT_PRIM_PATH}")

    added, replaced, skipped_no_visual = [], [], []
    for link in robot.GetChildren():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        visuals = link.GetChild("visuals")
        if not visuals.IsValid():
            skipped_no_visual.append(link.GetName())
            continue

        points, counts, indices = _copy_visual_mesh_to_link_space(link, visuals)
        if not points:
            skipped_no_visual.append(link.GetName())
            continue

        existing_collisions = link.GetChild("collisions")
        had_collision_before = existing_collisions.IsValid()
        if had_collision_before:
            # rebuild from scratch instead of layering on top of whatever shapes/approximation the
            # source asset already authored there (e.g. the feet's hand-placed contact spheres)
            stage.RemovePrim(existing_collisions.GetPath())

        # copy the link's own visual mesh into a fresh (non-instanced) collision mesh, and use PhysX's
        # SDF (signed distance field) approximation at cook time - unlike convex decomposition, this
        # keeps concave/thin detail (finger gaps, etc.) close to the real mesh, and unlike a raw
        # triangle mesh ("none"), it still generates contacts between two dynamic bodies (PhysX does
        # not support dynamic-vs-dynamic triangle mesh contacts at all).
        coll_xform = UsdGeom.Xform.Define(stage, link.GetPath().AppendChild("collisions"))
        # the source asset's own "collisions"/"visuals" prims are instanceable (via a payload arc from
        # configuration/*.usd); a fresh Xform.Define() here doesn't clear that composed opinion by
        # itself, so authoring children under it fails with "authoring to an instance proxy is not
        # allowed" unless we explicitly override instanceable=False on our own new prim spec.
        coll_xform.GetPrim().SetInstanceable(False)
        coll_mesh = UsdGeom.Mesh.Define(stage, coll_xform.GetPath().AppendChild("mesh"))
        coll_mesh.CreatePointsAttr(points)
        coll_mesh.CreateFaceVertexCountsAttr(counts)
        coll_mesh.CreateFaceVertexIndicesAttr(indices)
        # keep the render clean: this mesh is a physics-only proxy, the real mesh already renders via "visuals"
        UsdGeom.Imageable(coll_mesh.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(coll_mesh.GetPrim())
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(coll_mesh.GetPrim())
        mesh_collision_api.CreateApproximationAttr().Set("sdf")
        coll_mesh.GetPrim().AddAppliedSchema(_PHYSX_SDF_MESH_COLLISION_API)

        (replaced if had_collision_before else added).append(link.GetName())

    print(f"replaced existing collision ({len(replaced)}): {replaced}")
    print(f"added new collision ({len(added)}): {added}")
    if skipped_no_visual:
        print(f"skipped, no visual geometry found ({len(skipped_no_visual)}): {skipped_no_visual}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(OUT))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
