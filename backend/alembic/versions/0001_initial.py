"""Initial schema: patients (+ separate PII), screenings, captures, versioned results, reports,
calibration records and the audit log.

Revision ID: 0001
Revises:
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts(name: str, *, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.create_table(
        "patients",
        sa.Column("pseudonymous_id", sa.String(length=64), nullable=False),
        _ts("created_at"),
        sa.Column("age_years", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("pseudonymous_id", name=op.f("pk_patients")),
    )

    # PII lives only here, linked by the pseudonymous id.
    op.create_table(
        "patient_identities",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("pseudonymous_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("external_id", sa.String(length=100), nullable=True),
        _ts("created_at"),
        sa.ForeignKeyConstraint(
            ["pseudonymous_id"],
            ["patients.pseudonymous_id"],
            name=op.f("fk_patient_identities_pseudonymous_id_patients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patient_identities")),
        sa.UniqueConstraint("pseudonymous_id", name=op.f("uq_patient_identities_pseudonymous_id")),
    )

    op.create_table(
        "screenings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("pseudonymous_id", sa.String(length=64), nullable=False),
        _ts("created_at"),
        _ts("captured_at"),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("data_provenance", sa.String(length=16), nullable=False),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("screening_site_name", sa.String(length=200), nullable=True),
        sa.Column("camp_id", sa.String(length=100), nullable=True),
        sa.Column("health_worker_id", sa.String(length=100), nullable=True),
        sa.Column("health_worker_name", sa.String(length=200), nullable=True),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("consent", sa.JSON(), nullable=False),
        sa.Column("risk_factors", sa.JSON(), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("site_inputs", sa.JSON(), nullable=False),
        sa.Column("calibration_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["pseudonymous_id"],
            ["patients.pseudonymous_id"],
            name=op.f("fk_screenings_pseudonymous_id_patients"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_screenings")),
    )
    op.create_index(op.f("ix_screenings_created_at"), "screenings", ["created_at"])
    op.create_index(op.f("ix_screenings_pseudonymous_id"), "screenings", ["pseudonymous_id"])

    op.create_table(
        "captures",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("screening_id", sa.String(length=36), nullable=False),
        sa.Column("site", sa.String(length=48), nullable=True),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("file_name", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        _ts("created_at"),
        sa.ForeignKeyConstraint(
            ["screening_id"],
            ["screenings.id"],
            name=op.f("fk_captures_screening_id_screenings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_captures")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_captures_storage_key")),
    )
    op.create_index(op.f("ix_captures_screening_id"), "captures", ["screening_id"])

    op.create_table(
        "analysis_results",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("screening_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        _ts("created_at"),
        sa.Column("overall_risk_level", sa.String(length=16), nullable=False),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False),
        sa.Column("rule_config_version", sa.String(length=32), nullable=False),
        sa.Column("rule_config_sha256", sa.String(length=64), nullable=False),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["screening_id"],
            ["screenings.id"],
            name=op.f("fk_analysis_results_screening_id_screenings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analysis_results")),
        sa.UniqueConstraint(
            "screening_id", "version", name=op.f("uq_analysis_results_screening_id_version")
        ),
    )
    op.create_index(op.f("ix_analysis_results_screening_id"), "analysis_results", ["screening_id"])

    op.create_table(
        "reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("screening_id", sa.String(length=36), nullable=False),
        sa.Column("analysis_result_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        _ts("created_at"),
        sa.ForeignKeyConstraint(
            ["analysis_result_id"],
            ["analysis_results.id"],
            name=op.f("fk_reports_analysis_result_id_analysis_results"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["screening_id"],
            ["screenings.id"],
            name=op.f("fk_reports_screening_id_screenings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
        sa.UniqueConstraint(
            "analysis_result_id", "kind", name=op.f("uq_reports_analysis_result_id_kind")
        ),
    )
    op.create_index(op.f("ix_reports_screening_id"), "reports", ["screening_id"])

    op.create_table(
        "calibration_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("device_id", sa.String(length=100), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        _ts("created_at"),
        _ts("expires_at", nullable=True),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_calibration_records")),
    )
    op.create_index(
        op.f("ix_calibration_records_device_id"), "calibration_records", ["device_id"]
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        _ts("at"),
        sa.Column("actor", sa.String(length=100), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index(op.f("ix_audit_events_action"), "audit_events", ["action"])
    op.create_index(op.f("ix_audit_events_at"), "audit_events", ["at"])
    op.create_index("ix_audit_events_entity", "audit_events", ["entity_type", "entity_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_entity", table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_at"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_action"), table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index(op.f("ix_calibration_records_device_id"), table_name="calibration_records")
    op.drop_table("calibration_records")
    op.drop_index(op.f("ix_reports_screening_id"), table_name="reports")
    op.drop_table("reports")
    op.drop_index(op.f("ix_analysis_results_screening_id"), table_name="analysis_results")
    op.drop_table("analysis_results")
    op.drop_index(op.f("ix_captures_screening_id"), table_name="captures")
    op.drop_table("captures")
    op.drop_index(op.f("ix_screenings_pseudonymous_id"), table_name="screenings")
    op.drop_index(op.f("ix_screenings_created_at"), table_name="screenings")
    op.drop_table("screenings")
    op.drop_table("patient_identities")
    op.drop_table("patients")
