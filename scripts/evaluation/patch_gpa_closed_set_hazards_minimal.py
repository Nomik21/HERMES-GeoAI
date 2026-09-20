from pathlib import Path

PROJECT_ROOT = Path.cwd()
target = PROJECT_ROOT / "src" / "hermes" / "core" / "geo_perception.py"

text = target.read_text(encoding="utf-8")
backup = target.with_suffix(".py.bak_before_closedset_gpa")
backup.write_text(text, encoding="utf-8")

# 1. Replace open-set hazard list with six-class benchmark label space.
old_hazards = '''    HAZARDS = [
        "hurricane",
        "flood",
        "wildfire",
        "earthquake",
        "tornado",
        "landslide",
        "other",
    ]'''

new_hazards = '''    HAZARDS = [
        "drought",
        "earthquake",
        "flood",
        "hurricane",
        "ice_storm",
        "wildfire",
    ]'''

if old_hazards not in text:
    raise SystemExit("Could not find old HAZARDS block.")

text = text.replace(old_hazards, new_hazards)


# 2. Replace hazard description in the prompt.
old_description = '''        hazard_description = "\\n".join(
            [
                "Supported hazard types:",
                "- hurricane: tropical cyclone, storm surge, strong wind and rain;",
                "- flood: river flood, urban flood, inundation, standing water;",
                "- wildfire: forest fire, bushfire, open fire with smoke plumes;",
                "- earthquake: collapsed buildings, ground rupture, rubble without clear fire or flood;",
                "- tornado: narrow corridor wind damage, twisted debris, funnel cloud;",
                "- landslide: slope failure, soil or rock movement, buried roads or houses;",
                "- other: if none of the above clearly fits.",
            ]
        )'''

new_description = '''        hazard_description = "\\n".join(
            [
                "Supported hazard types are CLOSED-SET benchmark labels. Choose exactly one:",
                "- drought: dry land, drought stress, lack of water, dry vegetation, drought-related visual evidence;",
                "- earthquake: collapsed buildings, cracked structures, rubble, seismic structural damage;",
                "- flood: river flood, urban flood, inundation, standing water, submerged roads or buildings;",
                "- hurricane: tropical cyclone, storm surge, severe wind/rain damage, coastal storm impacts;",
                "- ice_storm: ice accumulation, snow/ice damage, frozen roads, winter-storm related impacts;",
                "- wildfire: forest fire, bushfire, burned vegetation/buildings, smoke, active or recent fire damage;",
            ]
        )'''

if old_description not in text:
    raise SystemExit("Could not find old hazard_description block.")

text = text.replace(old_description, new_description)


# 3. Make the instruction explicitly closed-set.
text = text.replace(
    '"2) Identify the most likely disaster type;\\n"',
    'f"2) Identify the most likely disaster type by choosing exactly ONE label from this closed set only: {\\\', \\\'.join(self.HAZARDS)};\\\\n"'
)

# The previous replacement uses an f-string inside string assembly; restore valid syntax manually.
text = text.replace(
    'f"2) Identify the most likely disaster type by choosing exactly ONE label from this closed set only: {\\\', \\\'.join(self.HAZARDS)};\\\\n"',
    'f"2) Identify the most likely disaster type by choosing exactly ONE label from this closed set only: {\', \'.join(self.HAZARDS)};\\\\n"'
)


# 4. Add deterministic closed-set enforcement methods after __init__.
insert_after = '''    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
'''

methods = '''
    def _normalize_hazard_label(self, value: Any) -> str:
        """
        Normalize aliases to the six benchmark hazard labels.
        """
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")

        aliases = {
            "drought": "drought",
            "dryness": "drought",
            "water_scarcity": "drought",

            "earthquake": "earthquake",
            "quake": "earthquake",
            "seismic": "earthquake",

            "flood": "flood",
            "flooded": "flood",
            "flooding": "flood",
            "inundation": "flood",
            "urban_flood": "flood",

            "hurricane": "hurricane",
            "tropical_cyclone": "hurricane",
            "cyclone": "hurricane",
            "typhoon": "hurricane",
            "storm_surge": "hurricane",
            "windstorm": "hurricane",
            "tornado": "hurricane",

            "ice_storm": "ice_storm",
            "icestorm": "ice_storm",
            "winter_storm": "ice_storm",
            "snowstorm": "ice_storm",
            "ice": "ice_storm",

            "wildfire": "wildfire",
            "fire": "wildfire",
            "forest_fire": "wildfire",
            "bushfire": "wildfire",
            "smoke": "wildfire",
        }

        return aliases.get(text, text)

    def _enforce_closed_set_hazard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce hazard_type ∈ HAZARDS while preserving the raw model output.
        """
        raw_hazard_type = data.get("hazard_type", "")
        normalized = self._normalize_hazard_label(raw_hazard_type)

        data["raw_hazard_type"] = raw_hazard_type
        data["allowed_hazard_labels"] = list(self.HAZARDS)
        data["closed_set_applied"] = False

        if normalized in self.HAZARDS:
            data["hazard_type"] = normalized
            data["closed_set_applied"] = normalized != str(raw_hazard_type).strip().lower()
            return data

        probs = data.get("hazard_probs", {})
        if isinstance(probs, dict):
            normalized_probs = {}

            for key, value in probs.items():
                label = self._normalize_hazard_label(key)
                if label not in self.HAZARDS:
                    continue

                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue

                normalized_probs[label] = normalized_probs.get(label, 0.0) + max(score, 0.0)

            if normalized_probs:
                data["hazard_type"] = max(normalized_probs.items(), key=lambda kv: kv[1])[0]
                data["hazard_probs"] = normalized_probs
                data["closed_set_applied"] = True
                data["closed_set_repair_reason"] = "invalid_raw_hazard_replaced_by_max_closed_set_probability"
                return data

        raise ValueError(
            "GPA produced hazard_type outside the six-class benchmark label space: "
            f"{raw_hazard_type!r}. Allowed labels: {self.HAZARDS}"
        )

'''

if insert_after not in text:
    raise SystemExit("Could not find ModePerceiver.__init__ insertion point.")

if "_enforce_closed_set_hazard" not in text:
    text = text.replace(insert_after, insert_after + methods)


# 5. Add hazard_probs default and enforce after model output.
old_defaults = '''        data.setdefault("hazard_type", "other")
        data.setdefault("hazard_confidence", 0.0)
        n_img = len(pil_images)'''

new_defaults = '''        data.setdefault("hazard_type", "")
        data.setdefault("hazard_confidence", 0.0)
        data.setdefault("hazard_probs", {})

        data = self._enforce_closed_set_hazard(data)

        n_img = len(pil_images)'''

if old_defaults not in text:
    raise SystemExit("Could not find old hazard default block.")

text = text.replace(old_defaults, new_defaults)


# 6. Strengthen final prompt constraint.
old_final = '''            "Do NOT include explanations outside the JSON. JSON must be valid and parseable."
        )'''

new_final = '''            "hazard_type MUST be one of: drought, earthquake, flood, hurricane, ice_storm, wildfire. "
            "Do NOT output tornado, landslide, other, or any label outside the six-label benchmark set. "
            "Do NOT include explanations outside the JSON. JSON must be valid and parseable."
        )'''

if old_final not in text:
    raise SystemExit("Could not find final JSON instruction block.")

text = text.replace(old_final, new_final)

target.write_text(text, encoding="utf-8")

print(f"Patched: {target}")
print(f"Backup written to: {backup}")
print("GPA hazard label space is now six-class closed-set ✅")
