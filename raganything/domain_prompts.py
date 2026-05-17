from __future__ import annotations

from copy import deepcopy
from typing import Any

from lightrag.prompt import PROMPTS


WEAPON_EQUIPMENT_ENTITY_TYPES = [
    "ScientificPrinciple",
    "ApplicationTechnology",
    "SystemEffect",
]

_ORIGINAL_PROMPTS: dict[str, Any] | None = None


def _snapshot_original_prompts() -> None:
    global _ORIGINAL_PROMPTS
    if _ORIGINAL_PROMPTS is None:
        _ORIGINAL_PROMPTS = deepcopy(PROMPTS)


def reset_domain_prompts() -> None:
    """Restore LightRAG's default extraction prompts in this Python process."""
    if _ORIGINAL_PROMPTS is None:
        return
    PROMPTS.clear()
    PROMPTS.update(deepcopy(_ORIGINAL_PROMPTS))


def apply_weapon_equipment_domain_prompts() -> None:
    """Use a weapon-equipment R&D concept graph extraction profile.

    This intentionally replaces LightRAG's generic entity extraction prompts while
    keeping the output delimiters and record schema that LightRAG's parser needs.
    """
    _snapshot_original_prompts()

    PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a knowledge graph specialist for weapon-equipment research and development.
Your task is to extract strictly grounded, non-operational scientific principles,
application technologies, system effects, and their high-value conceptual links.

---Entity Types---
Extract only the following three entity types. Do not create any other type.

1. ScientificPrinciple
Scientific principle layer: a concrete natural-science principle, law, effect,
mechanism, or material/field interaction that belongs to a discipline such as
electromagnetics, thermodynamics, solid-state physics, fluid mechanics, acoustics,
materials chemistry, optics, mechanics, or control theory. The entity name must be
a recognizable scientific/engineering principle, not an ability, mission objective,
requirement, strategy, or abstract concept. The description must include
`discipline=...`.

2. ApplicationTechnology
Application technology layer: a physical-world technology, device, component,
material structure, sensor, actuator, algorithmic control module tied to hardware,
or engineering implementation based on one or two scientific principles. It must
be something that can be designed, fabricated, integrated, measured, or deployed.
Do not use a vague technology direction, program name, mission concept, capability
bundle, or "X system integration" phrase as an entity. The description must include
`domain=military/civilian/dual-use` and `based_on=...`. If the source text does not
explicitly identify the underlying principles, write `based_on=not explicit`.

3. SystemEffect
System effect layer: a physical-world observable or measurable system-level effect,
state, signal change, performance index, or interaction result produced by one
technology or a combination of technologies in an application context. It must be
expressible as a measurable effect on energy, waves, motion, structure, signal,
thermal state, detection signature, stability, accuracy, latency, range, reliability,
or similar physical/performance quantities. Do not extract slogans, generalized
abilities, operational styles, task goals, or broad phrases such as "隐蔽快速机动",
"系统机动集成", "定向能防空武器系统机动集成", "作战效能提升", or "能力增强". The
description must include `scenario=...` and `performance=...`.

---Relationship Types---
Extract only the following relationships between extracted entities.

1. VerticalEnable
Vertical enabling relationship:
- ScientificPrinciple -> ApplicationTechnology: a principle enables a technology.
- ApplicationTechnology -> SystemEffect: a technology achieves an effect.

2. HorizontalRelation
In-layer horizontal relationship:
- ScientificPrinciple <-> ScientificPrinciple: complement, analogy, or conflict.
- ApplicationTechnology <-> ApplicationTechnology: synergy, substitute,
  complement, or conflict.
- SystemEffect <-> SystemEffect: synergy, tradeoff, or conflict.

The relationship type and subtype must be written in `relationship_keywords`, for
example:
VerticalEnable, principle_to_technology
VerticalEnable, technology_to_effect
HorizontalRelation, complement
HorizontalRelation, analogy
HorizontalRelation, conflict
HorizontalRelation, synergy
HorizontalRelation, substitute
HorizontalRelation, tradeoff

---Extraction Rules---
1. Extract only concepts that appear in the text or are directly supported by it.
   Do not invent new concepts or speculative weapon designs.
2. Hard budget: extract at most 4 entity records from each input chunk. If there
   are more than 4 candidates, keep only the most explicit, central, and reusable
   candidates, and discard weak or generic candidates. It is valid to output fewer
   than 4 entities, or no entities, when the chunk lacks qualifying content.
3. Hard type constraint: entity_type must be exactly one of ScientificPrinciple,
   ApplicationTechnology, or SystemEffect. Never output any other entity type.
4. Soft layer balance: when the chunk supports several candidates, prefer a compact
   cross-layer set that explains a principle -> technology -> effect chain. Do not
   fill the 4-entity budget with marginal same-layer variants.
5. Entity names must denote things with a clear real-world referent or accepted
   scientific referent:
   - ScientificPrinciple: natural-science or engineering principle/effect/mechanism.
   - ApplicationTechnology: physical technology, device, component, material
     structure, or hardware-bound control module.
   - SystemEffect: observable or measurable system effect/performance quantity.
6. Do not extract generic words or generic phrases such as technology, method,
   system, equipment, capability, scheme, performance, integration, mobility,
   stealth, rapid response, protection, superiority, optimization, effectiveness,
   or "X能力" unless the phrase names a specific physical technology or measurable
   effect stated in the source text.
7. Do not extract combined mission/capability phrases, project themes, or sentence
   fragments as entities. If a phrase cannot answer "what physical principle,
   physical technology, or measurable physical/system effect is this?", skip it.
8. Do not extract concrete manufacturing steps, executable use procedures,
   attack tactics, parameter recipes, process settings, or operational advice.
9. Entity names must be stable and reusable. If both an abbreviation and full name
   appear, prefer the full name and mention the abbreviation in the description.
10. If a concept could belong to multiple layers, prefer the most fundamental layer:
   ScientificPrinciple before ApplicationTechnology before SystemEffect.
11. Relationships must connect entities that are already extracted.
12. Treat horizontal relationships as undirected unless the source text explicitly
   states direction.
13. Entity names, descriptions, tag values, and relationship descriptions must be
   written in Chinese. Translate English technical terms into their common Chinese
   names whenever a common Chinese name exists.
14. If an English abbreviation or original term is important for identification,
   put it in parentheses in the description, not as the primary entity name.
   Example: use `光子晶体带隙调控` as entity_name and mention `Photonic Crystal
   Bandgap Control` in the description.
15. Keep only schema labels unchanged: entity_type values, VerticalEnable,
   HorizontalRelation, and subtype keywords must remain in English because the
   parser and visualization use these labels.

---Output Format---
Use exactly one record per line.

Entity records have exactly 4 fields:
entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description

Relationship records have exactly 5 fields:
relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description

The only allowed entity types are: {entity_types}.
Output at most 4 entity records. Output all entity records first, then relationship records.
End with the literal completion marker {completion_delimiter}.

---Examples---
{examples}
"""

    PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract the weapon-equipment R&D concept graph entities and relationships from the
input text.

---Allowed Entity Types---
[{entity_types}]

---Data to be Processed---
<Input Text>
```
{input_text}
```

<Output>
"""

    PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Review the previous extraction for the same input text and output only missing or
incorrectly formatted entities and relationships that satisfy the weapon-equipment
R&D concept graph schema. Do not cause the total extraction for this chunk to
exceed 4 entity records.

---Allowed Entity Types---
[{entity_types}]

---Data to be Processed---
<Input Text>
```
{input_text}
```

<Output>
"""

    PROMPTS["entity_extraction_examples"] = [
        """Example 1:

Input:
Magnetostrictive materials change shape under an external magnetic field. A
magnetostrictive actuator can use this effect for precise displacement control,
which supports millisecond-level angular-position error suppression on
high-dynamic stabilized platforms.

Output:
entity{tuple_delimiter}磁致伸缩效应{tuple_delimiter}ScientificPrinciple{tuple_delimiter}discipline=固体物理；磁致伸缩效应描述磁性材料在外加磁场作用下发生形状或尺寸变化的物理现象。
entity{tuple_delimiter}磁致伸缩驱动器{tuple_delimiter}ApplicationTechnology{tuple_delimiter}domain=军民两用；based_on=磁致伸缩效应；磁致伸缩驱动器利用磁场诱导材料形变实现精密位移控制。
entity{tuple_delimiter}角位置误差抑制{tuple_delimiter}SystemEffect{tuple_delimiter}scenario=高动态平台稳定控制；performance=毫秒级响应下减小平台角位置偏差。
relation{tuple_delimiter}磁致伸缩效应{tuple_delimiter}磁致伸缩驱动器{tuple_delimiter}VerticalEnable, principle_to_technology{tuple_delimiter}磁致伸缩效应为磁致伸缩驱动器提供磁场诱导形变的物理机制。
relation{tuple_delimiter}磁致伸缩驱动器{tuple_delimiter}角位置误差抑制{tuple_delimiter}VerticalEnable, technology_to_effect{tuple_delimiter}磁致伸缩驱动器的快速位移输出可支撑角位置误差抑制这一可测量系统效果。
{completion_delimiter}""",
        """Example 2:

Input:
Photonic crystal bandgap control and acoustic metamaterial local resonance are
analogous because both regulate wave propagation through engineered periodic or
resonant structures. Photonic crystal filters and acoustic metamaterial absorbers
can be combined in a platform protection concept to reduce radar cross section and
acoustic scattering intensity.

Output:
entity{tuple_delimiter}光子晶体带隙调控{tuple_delimiter}ScientificPrinciple{tuple_delimiter}discipline=电磁学；光子晶体带隙调控通过人工周期结构调节电磁波传播行为，英文术语为Photonic Crystal Bandgap Control。
entity{tuple_delimiter}声学超材料局域共振{tuple_delimiter}ScientificPrinciple{tuple_delimiter}discipline=声学；声学超材料局域共振通过设计的共振结构调节声波响应，英文术语为Acoustic Metamaterial Local Resonance。
entity{tuple_delimiter}光子晶体滤波器{tuple_delimiter}ApplicationTechnology{tuple_delimiter}domain=军民两用；based_on=光子晶体带隙调控；光子晶体滤波器利用带隙行为选择性控制电磁波透射。
entity{tuple_delimiter}雷达散射截面降低{tuple_delimiter}SystemEffect{tuple_delimiter}scenario=平台电磁特征控制；performance=降低目标对入射电磁波的等效散射面积。
relation{tuple_delimiter}光子晶体带隙调控{tuple_delimiter}声学超材料局域共振{tuple_delimiter}HorizontalRelation, analogy{tuple_delimiter}两类原理都通过人工结构调控波传播行为，在电磁和声学领域之间形成类比关系。
relation{tuple_delimiter}光子晶体滤波器{tuple_delimiter}雷达散射截面降低{tuple_delimiter}VerticalEnable, technology_to_effect{tuple_delimiter}光子晶体滤波器通过电磁波传播控制支撑雷达散射截面降低。
{completion_delimiter}""",
    ]


def configure_extraction_profile(profile: str | None) -> dict[str, Any]:
    """Apply a named extraction profile and return LightRAG addon overrides."""
    normalized = (profile or "default").strip().lower()
    if normalized in {"", "default", "generic", "lightrag"}:
        reset_domain_prompts()
        return {}
    if normalized in {"weapon_equipment", "weapon-equipment", "weapons"}:
        apply_weapon_equipment_domain_prompts()
        return {"entity_types": WEAPON_EQUIPMENT_ENTITY_TYPES, "language": "Chinese"}
    raise ValueError(
        f"Unknown extraction_profile '{profile}'. Supported values: default, weapon_equipment"
    )
