"""The tutorial project: a small, complete TVC rocket.

Everything here is built through the same public APIs the UI uses — parametric features,
assembly instances, mate constraints, holes, fasteners and hardware definitions — so the
sample doubles as an integration test of the whole chain.

Geometry: rocket axis along +Z, nozzle exit near z = 0. The airframe is grounded, the gimbal
cage and motor case are hinged to it, and two servo components drive the hinges. The motor
uses the synthetic DEMO thrust curve (labelled as such) until real data is imported.
"""

from __future__ import annotations

import math

from ..assembly.constraints import AssemblyConstraint, ConstraintType, EntityKind, EntityRef
from ..assembly.model import Assembly
from ..cad.document import PartRepository
from ..cad.features import (
    BoxFeature,
    ConeFeature,
    CylinderFeature,
    HoleFeature,
    PatternPolarFeature,
    TubeFeature,
)
from ..components.properties import MechanicalProperties, PhysicalProperties
from ..core.math3d import Transform, vec3
from ..fasteners.catalog import FastenerKind, metric_thread
from ..fasteners.install import InstalledFastener
from ..physics.aero import AeroConfig, AeroModelKind
from ..physics.flight import FlightConfig
from .schema import Project


def _make_part(repo: PartRepository, name: str, features: list, material: str, color: str,
               notes: str = ""):
    part = repo.create(name, features, material_id=material)
    part.color = color
    part.notes = notes
    part.physical.material_id = material
    return part


def build_sample_project(name: str = "TVC Demo Rocket") -> Project:
    project = Project(name=name)
    repo = project.parts
    materials = project.materials
    asm = project.assembly
    asm.name = "TVC rocket assembly"

    # ------------------------------------------------------------------ parts
    airframe = _make_part(
        repo, "Airframe tube",
        [TubeFeature(inner_radius=0.019, outer_radius=0.0205, height=0.280,
                     origin=(0, 0, 0.072), name="Airframe 40 mm tube"),
         ],
        "pla", "#9fb3c8",
        "Main 40 mm airframe tube. Holes are modelled for the fin and bulkhead fasteners.",
    )
    nose = _make_part(
        repo, "Nose cone",
        [ConeFeature(radius1=0.0205, radius2=0.0015, height=0.095, origin=(0, 0, 0.352),
                     name="Ogive-ish nose (conical)"),
         CylinderFeature(radius=0.0188, height=0.012, origin=(0, 0, 0.340),
                         operation="join", name="Nose shoulder")],
        "pla", "#c0d0e0",
        "Conical nose with a shoulder. Replace with an ogive profile in the CAD workspace.",
    )
    fins = _make_part(
        repo, "Fin set",
        [BoxFeature(0.090, 0.0015, 0.070, origin=(0.0205, -0.00075, 0.055), name="Fin blank"),
         PatternPolarFeature(count=4, axis_origin=(0, 0, 0.090), axis_dir=(0, 0, 1),
                             name="4-fin pattern")],
        "g10_fr4", "#6f8f6f",
        "Four 1.5 mm G10 fins (60 mm span, 60 mm root chord), patterned from one blank.",
    )
    mount = _make_part(
        repo, "Gimbal mount",
        [BoxFeature(0.048, 0.048, 0.010, origin=(0, 0, 0.062), centered=False, name="Mount plate"),
         HoleFeature(position=(0.0, 0.0, 0.072), axis=(0, 0, -1), diameter=0.030, depth=0.010,
                     through=True, pre_threaded=False, name="Gimbal clearance bore"),
         HoleFeature(position=(0.018, 0.018, 0.072), axis=(0, 0, -1), diameter=0.0025, depth=0.010,
                     through=True, pre_threaded=True, name="M3 hole 1"),
         HoleFeature(position=(-0.018, 0.018, 0.072), axis=(0, 0, -1), diameter=0.0025, depth=0.010,
                     through=True, pre_threaded=True, name="M3 hole 2"),
         HoleFeature(position=(-0.018, -0.018, 0.072), axis=(0, 0, -1), diameter=0.0025, depth=0.010,
                     through=True, pre_threaded=True, name="M3 hole 3"),
         HoleFeature(position=(0.018, -0.018, 0.072), axis=(0, 0, -1), diameter=0.0025, depth=0.010,
                     through=True, pre_threaded=True, name="M3 hole 4")],
        "al_6061_t6", "#b0b6bc",
        "Aluminium gimbal mount, 4 x M3 threaded holes, central clearance bore for the cage.",
    )
    cage = _make_part(
        repo, "Gimbal cage",
        [TubeFeature(inner_radius=0.016, outer_radius=0.0205, height=0.030,
                     origin=(0, 0, 0.030), name="Cage ring"),
         BoxFeature(0.004, 0.030, 0.012, origin=(0.0185, -0.015, 0.040), name="Cage arm")],
        "pla", "#d0a060",
        "TVC cage: carries the motor, hinged to the mount. Servo drive is direct on the hinge.",
    )
    motor_case = _make_part(
        repo, "Motor case",
        [TubeFeature(inner_radius=0.0130, outer_radius=0.0145, height=0.072,
                     origin=(0, 0, 0.012), name="Motor case tube")],
        "al_7075_t6", "#7a7f85",
        "Motor case tube (1.5 mm wall). Propellant mass is added by the motor definition.",
    )
    nozzle = _make_part(
        repo, "Nozzle",
        [ConeFeature(radius1=0.012, radius2=0.008, height=0.018, origin=(0, 0, 0.002),
                     name="Nozzle bell")],
        "stainless_304", "#8a8a8a",
        "Nozzle; its centre is the gimbal/nozzle reference for thrust application.",
    )
    bulkhead = _make_part(
        repo, "Bulkhead",
        [CylinderFeature(radius=0.019, height=0.006, origin=(0, 0, 0.140), name="Bulkhead disc"),
         HoleFeature(position=(0, 0, 0.146), axis=(0, 0, -1), diameter=0.008, depth=0.006,
                     through=True, pre_threaded=False, name="Wire pass-through"),
         HoleFeature(position=(0.013, 0, 0.146), axis=(0, 0, -1), diameter=0.0032, depth=0.006,
                     through=True, pre_threaded=False, name="Avionics screw hole 1"),
         HoleFeature(position=(-0.0065, 0.0113, 0.146), axis=(0, 0, -1), diameter=0.0032, depth=0.006,
                     through=True, pre_threaded=False, name="Avionics screw hole 2"),
         HoleFeature(position=(-0.0065, -0.0113, 0.146), axis=(0, 0, -1), diameter=0.0032, depth=0.006,
                     through=True, pre_threaded=False, name="Avionics screw hole 3")],
        "pla", "#a8c0d8",
        "Avionics bulkhead with a wire pass-through and three M3 clearance holes.",
    )
    servo_part = _make_part(
        repo, "MG90S servo",
        [BoxFeature(0.0228, 0.0122, 0.0285, origin=(0, 0, 0.0), name="Servo housing"),
         CylinderFeature(radius=0.004, height=0.004, origin=(0, -0.0061, 0.0142), name="Output boss")],
        "pla", "#3d6fa5",
        "Servo envelope with output boss. Physical parameters come from the MG90S hardware definition.",
    )
    pico_part = _make_part(
        repo, "Pico 2 W",
        [BoxFeature(0.051, 0.021, 0.005, origin=(0, 0, 0.0), name="Board outline")],
        "g10_fr4", "#2f7f4f",
        "Raspberry Pi Pico 2 W board outline (51 x 21 x 5 mm envelope).",
    )
    imu_part = _make_part(
        repo, "MPU6050",
        [BoxFeature(0.020, 0.015, 0.003, origin=(0, 0, 0.0), name="IMU board")],
        "g10_fr4", "#28507a",
        "MPU6050 breakout. Its mounting pose defines the simulated sensor location and axes.",
    )

    # ------------------------------------------------------------- assembly
    airframe_inst = asm.create_component(airframe.id, "Airframe", position=(0, 0, 0))
    airframe_inst.grounded = True
    airframe_inst.physical.material_id = "pla"
    nose_inst = asm.create_component(nose.id, "Nose cone", parent_id=airframe_inst.id, position=(0, 0, 0))
    nose_inst.physical.material_id = "pla"
    fins_inst = asm.create_component(fins.id, "Fins", parent_id=airframe_inst.id, position=(0, 0, 0))
    fins_inst.physical.material_id = "g10_fr4"
    fins_inst.metadata["role"] = "fin"
    fins_inst.metadata["fin_count"] = 4
    fins_inst.metadata["fin_root_chord"] = 0.070
    fins_inst.metadata["fin_tip_chord"] = 0.045
    fins_inst.metadata["fin_sweep"] = 0.0
    bulkhead_inst = asm.create_component(bulkhead.id, "Bulkhead", parent_id=airframe_inst.id, position=(0, 0, 0))
    bulkhead_inst.physical.material_id = "pla"
    mount_inst = asm.create_component(mount.id, "Gimbal mount", parent_id=airframe_inst.id, position=(0, 0, 0))
    mount_inst.physical.material_id = "al_6061_t6"
    cage_inst = asm.create_component(cage.id, "Gimbal cage", parent_id=airframe_inst.id, position=(0, 0, 0))
    cage_inst.physical.material_id = "pla"
    cage_inst.mechanical.backlash = math.radians(0.6)
    cage_inst.mechanical.damping = 0.004
    cage_inst.mechanical.friction_static = 0.004
    cage_inst.mechanical.friction_kinetic = 0.003
    cage_inst.metadata["role"] = "tvc"
    cage_inst.metadata["tvc_channel"] = "pitch"

    motor_inst = asm.create_component(motor_case.id, "Motor case", parent_id=cage_inst.id, position=(0, 0, 0))
    motor_inst.physical.material_id = "al_7075_t6"
    motor_inst.mechanical.backlash = math.radians(0.6)
    motor_inst.mechanical.damping = 0.004
    motor_inst.mechanical.friction_static = 0.004
    motor_inst.mechanical.friction_kinetic = 0.003
    motor_inst.metadata["role"] = "motor"
    motor_inst.metadata["tvc_channel"] = "yaw"

    nozzle_inst = asm.create_component(nozzle.id, "Nozzle", parent_id=motor_inst.id, position=(0, 0, 0))
    nozzle_inst.physical.material_id = "stainless_304"
    nozzle_inst.metadata["role"] = "nozzle"

    servo_pitch = asm.create_component(servo_part.id, "Pitch servo", parent_id=airframe_inst.id,
                                       position=(0.023, 0.0, 0.050))
    servo_pitch.physical.material_id = "pla"
    servo_pitch.hardware_type = "servo"
    servo_pitch.hardware_id = "mg90s"
    servo_pitch.metadata["servo_for"] = "pitch"
    servo_pitch.color = "#3d6fa5"

    servo_yaw = asm.create_component(servo_part.id, "Yaw servo", parent_id=airframe_inst.id,
                                     position=(-0.023, 0.0, 0.050))
    servo_yaw.physical.material_id = "pla"
    servo_yaw.hardware_type = "servo"
    servo_yaw.hardware_id = "mg90s"
    servo_yaw.metadata["servo_for"] = "yaw"
    servo_yaw.color = "#3d6fa5"

    pico_inst = asm.create_component(pico_part.id, "Flight computer", parent_id=airframe_inst.id,
                                     position=(0, 0, 0.270))
    pico_inst.physical.material_id = "g10_fr4"
    pico_inst.hardware_type = "board"
    pico_inst.hardware_id = "pico2w"
    pico_inst.metadata["role"] = "avionics"

    imu_inst = asm.create_component(imu_part.id, "IMU", parent_id=airframe_inst.id,
                                    position=(0, 0, 0.250))
    imu_inst.physical.material_id = "g10_fr4"
    imu_inst.hardware_type = "sensor"
    imu_inst.hardware_id = "mpu6050"
    imu_inst.metadata["role"] = "imu"

    # bind servos to the hinge joints through metadata
    cage_inst.metadata["servo_component_id"] = servo_pitch.id
    motor_inst.metadata["servo_component_id"] = servo_yaw.id

    # ---------------------------------------------------------- constraints
    asm.constraints.append(AssemblyConstraint(
        id="hinge_pitch",
        type=ConstraintType.REVOLUTE,
        name="Pitch hinge (servo-driven)",
        a=EntityRef(airframe_inst.id, EntityKind.AXIS, (0, 0, 0.045), (0, 1, 0), "pitch axis"),
        b=EntityRef(cage_inst.id, EntityKind.AXIS, (0, 0, 0.045), (0, 1, 0), "pitch axis"),
    ))
    asm.constraints.append(AssemblyConstraint(
        id="hinge_yaw",
        type=ConstraintType.REVOLUTE,
        name="Yaw hinge (servo-driven)",
        a=EntityRef(cage_inst.id, EntityKind.AXIS, (0, 0, 0.045), (1, 0, 0), "yaw axis"),
        b=EntityRef(motor_inst.id, EntityKind.AXIS, (0, 0, 0.045), (1, 0, 0), "yaw axis"),
    ))

    # ------------------------------------------------------------- fasteners
    mount_holes = repo.holes_for(mount.id, mount_inst.id)
    asm.add_holes(mount_holes)
    screw = next(
        s for s in project.fasteners.all()
        if s.kind is FastenerKind.MACHINE_SCREW and s.thread.designation == "M3"
        and abs(s.length - 0.008) < 1e-6 and s.head_type.value == "socket_head"
    )
    for index, hole in enumerate(mount_holes):
        if "M3" not in hole.label:
            continue
        fastener = InstalledFastener(
            id=f"screw_mount_{index+1}",
            fastener_id=screw.id,
            hole_id=hole.id,
            with_washer=True,
        )
        fastener.set_depth(0.006, screw.thread)
        fastener.installed = True
        asm.install(fastener)

    bulkhead_holes = repo.holes_for(bulkhead.id, bulkhead_inst.id)
    asm.add_holes(bulkhead_holes)
    for index, hole in enumerate(bulkhead_holes):
        if "screw" not in hole.label.lower():
            continue
        fastener = InstalledFastener(
            id=f"screw_bulkhead_{index+1}",
            fastener_id=screw.id,
            hole_id=hole.id,
            with_nut=True,
        )
        fastener.set_depth(screw.length, screw.thread)
        fastener.installed = True
        asm.install(fastener)

    # ------------------------------------------------------- electronics/physics
    project.electrical.imu_component_id = imu_inst.id
    project.electrical.board_id = "pico2w"
    project.electrical.pitch_servo_id = "mg90s"
    project.electrical.yaw_servo_id = "mg90s"
    project.electrical.supply_voltage = 5.0
    project.electrical.loop_rate_hz = 250.0
    project.control.max_gimbal = math.radians(8.0)
    project.control.tilt_kp = 2.2
    project.control.tilt_ki = 0.15
    project.control.tilt_kd = 0.08
    project.control.rate_damping = 0.18

    project.aero = AeroConfig(
        model=AeroModelKind.GEOMETRY_ESTIMATE,
        cd0=0.48,
        cn_alpha=2.0,
        reference_area=math.pi * (0.041 / 2.0) ** 2,
        reference_length=0.447,
        uncertainty_percent=25.0,
        notes="Generic coefficient estimate for a 41 mm airframe. Replace with measured data.",
    )
    project.flight = FlightConfig(
        dt=0.001, max_time=90.0, launch_elevation_deg=90.0, rail_length=1.2,
        parachute_area=0.6, deploy_at_apogee=True, max_tilt_deg=25.0,
    )
    project.simulation.mech_dt = 0.001
    project.active_motor_id = "demo_motor"
    project.component_library = [
        {
            "id": "lib_mg90s",
            "kind": "servo",
            "name": "MG90S (configured)",
            "data": project.hardware.servo("mg90s").to_dict(),
        },
        {
            "id": "lib_mpu6050",
            "kind": "sensor",
            "name": "MPU6050 (configured)",
            "data": project.hardware.sensor("mpu6050").to_dict(),
        },
        {
            "id": "lib_pico2w",
            "kind": "board",
            "name": "Pico 2 W (configured)",
            "data": project.hardware.board("pico2w").to_dict(),
        },
    ]
    project.notes = (
        "Tutorial project. Airframe grounded; gimbal cage and motor case on servo-driven hinges; "
        "gimbal mount has 4 installed M3 screws and the bulkhead 3 more. The motor uses the "
        "synthetic DEMO thrust curve — import a real .eng file before evaluating performance."
    )
    return project


def build_blank_project(name: str = "Untitled project") -> Project:
    """An empty project with libraries ready but no geometry."""
    project = Project(name=name)
    project.assembly.name = "Assembly 1"
    return project
