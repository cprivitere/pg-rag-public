"""Contract tests for the il2cpp decomp document builder.

Uses tmp dirs per TEST_CONTRACTS L7 — the real data/il2cpp tree is never read..
"""

import json

import pytest

from pgrag.documents import decomp_builder


def _build(tmp_path, monkeypatch, dump_cs="", stringlit=None):
    dump_dir = tmp_path / "out_lean" / "Dump0"
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "dump.cs").write_text(dump_cs, encoding="utf-8")
    if stringlit is not None:
        (dump_dir / "stringliteral.json").write_text(json.dumps(stringlit), encoding="utf-8")
    monkeypatch.setattr(decomp_builder, "IL2CPP_DIR", tmp_path)
    return decomp_builder.build_il2cpp_documents()


def test_missing_dump_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(decomp_builder, "IL2CPP_DIR", tmp_path)
    assert decomp_builder.build_il2cpp_documents() == []


def test_assembly_filter_excludes_engine_types(tmp_path, monkeypatch):
    dump = (
        "// Dll : System.Xml.dll\n"
        "public enum XmlThing\n"
        "{\n"
        "    public const XmlThing Zero = (XmlThing)0;\n"
        "    public int value__;\n"
        "}\n"
        "// Dll : Assembly-CSharp.dll\n"
        "public enum CameraMode\n"
        "{\n"
        "    public const CameraMode Game = defaultValue;\n"
        "    public int value__;\n"
        "}\n"
        "// Dll : GorgonCore.dll\n"
        "public enum SkillType\n"
        "{\n"
        "    public const SkillType Unknown = 0;\n"
        "}\n"
    )
    docs = _build(tmp_path, monkeypatch, dump_cs=dump)
    assert [d["id"] for d in docs] == ["il2cpp_enum_SkillType"]
    doc = docs[0]
    assert doc["type"] == "enum"
    assert doc["metadata"]["source"] == "il2cpp"
    assert doc["metadata"]["table"] == "enums"
    assert doc["metadata"]["name"] == "SkillType"
    assert "Unknown = 0" in doc["text"]


def test_schema_card_target_classes_only(tmp_path, monkeypatch):
    dump = (
        "// Dll : System.Xml.dll\n"
        "public class Item // TypeDefIndex: 100\n"
        "{\n"
        "    public string m_name; // 0x9\n"
        "    public int m_version; // 0x6\n"
        "}\n"
        "// Dll : GorgonCore.dll\n"
        "public class Item // TypeDefIndex: 200\n"
        "{\n"
        "    public string m_name; // 0x9\n"
        "    public int m_level; // 0x6\n"
        "    public void Load() { }\n"
        "}\n"
    )
    docs = _build(tmp_path, monkeypatch, dump_cs=dump)
    assert [d["id"] for d in docs] == ["il2cpp_schema_Item"]
    doc = docs[0]
    assert doc["type"] == "schema"
    text = doc["text"]
    assert "data model" in text
    assert "- public string m_name" in text
    assert "- public int m_level" in text
def test_area_schema_card_via_area_info(tmp_path, monkeypatch):
    dump = (
        "// Dll : GorgonCore.dll\n"
        "public class AreaInfo // TypeDefIndex: 300\n"
        "{\n"
        "    public int m_AreaID; // 0x4\n"
        "    public string DisplayName; // 0x5\n"
        "}\n"
        "public class Area // TypeDefIndex: 301\n"
        "{\n"
        "    public int m_Zone; // 0x7\n"
        "}\n"
    )
    docs = _build(tmp_path, monkeypatch, dump_cs=dump)
    assert [d["id"] for d in docs] == ["il2cpp_schema_AreaInfo"]
    doc = docs[0]
    assert doc["type"] == "schema"
    assert doc["metadata"]["table"] == "schema"
    text = doc["text"]
    assert "- public int m_AreaID" in text
    assert "- public string DisplayName" in text
    assert all("il2cpp_schema_Area" != d["id"] for d in docs)



def test_mechanic_allowlist_and_hygiene(tmp_path, monkeypatch):
    stringlit = [
        {"index": 0, "value": "When both of your combat bars are full, you gain extra XP toward the skills you used during the battle."},
        {"index": 1, "value": "short"},
        {"index": 2, "value": "<html>both of your combat bars and nonsense"},
        {"index": 3, "value": "{0} both of your combat bars and more nonsense"},
    ]
    docs = _build(tmp_path, monkeypatch, stringlit=stringlit)
    assert [d["id"] for d in docs] == ["il2cpp_mechanic_combat-xp-attribution_0"]
    assert len(docs) == 1
    assert docs[0]["type"] == "mechanic"
    assert docs[0]["metadata"]["table"] == "mechanic"
    text = docs[0]["text"]
    assert "both of your combat bars" in text
    assert "<" not in text
    assert "\n" not in text
    assert "{0}" not in text


def test_deterministic_ids_and_order(tmp_path, monkeypatch):
    dump = (
        "// Dll : GorgonCore.dll\n"
        "public enum SkillType\n"
        "{\n"
        "    public const SkillType Unknown = 0;\n"
        "    public const SkillType Melee Weapon = (SkillType)1;\n"
        "}\n"
        "public class Item // TypeDefIndex: 200\n"
        "{\n"
        "    public string m_name; // 0x9\n"
        "    public int m_level; // 0x6\n"
        "}\n"
    )
    stringlit = [
        {"index": 0, "value": "When both of your combat bars are full, you gain extra XP toward the skills you used during the battle."},
        {"index": 1, "value": "When both of your combat bars are full, you gain extra XP toward the skills you used during the battle."},
    ]
    first = _build(tmp_path, monkeypatch, dump_cs=dump, stringlit=stringlit)
    second = _build(tmp_path, monkeypatch, dump_cs=dump, stringlit=stringlit)
    first_keyed = [(d["id"], d["text"]) for d in first]
    second_keyed = [(d["id"], d["text"]) for d in second]
    assert first_keyed == second_keyed
    assert sum(1 for d in first if d["type"] == "mechanic") == 1  # dup normalized value collapses