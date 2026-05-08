"""
Pydantic schemas for structured extraction across three use cases.

Use Case 1: Counterfactual Mortality Estimation
Use Case 2: Sepsis Phenotype Extraction
Use Case 3: Biomarker Selection for Risk Stratification
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class SourceAnchor(BaseModel):
    """Links an extracted value back to the original document."""

    page_number: int = Field(
        description="Page number in the source PDF where the value was found."
    )
    quote: str = Field(
        description=(
            "Exact sentence or short passage from the source text that "
            "supports the extracted value. Use 'Not reported' if unavailable."
        )
    )
    section_name: str = Field(
        default="Not reported",
        description=(
            "Section where evidence was found (e.g., Abstract, Methods, Results)."
        ),
    )
    evidence_origin: str = Field(
        default="raw_text",
        description=(
            "Origin of evidence text: raw_text, vlm_summary, or mixed."
        ),
    )


# ---------------------------------------------------------------------------
# Use Case 1 – Counterfactual Mortality Estimation
# ---------------------------------------------------------------------------

class UseCase1Evidence(BaseModel):
    """
    Structured evidence for counterfactual mortality estimation.

    Captures predictor–outcome associations, effect sizes, and statistical
    context so that downstream models can estimate expected mortality without
    a control group.
    """

    study_name: str = Field(
        description="Author surname(s) and year, e.g. 'Leona 2025'."
    )
    population: str = Field(
        description="Patient population described in the study."
    )
    sample_size: str = Field(
        description=(
            "Total number of patients (N). Write 'Not reported' if not stated."
        )
    )
    setting: str = Field(
        description=(
            "Clinical setting, e.g. ICU, ED, ward. Write 'Not reported' if "
            "not stated."
        )
    )
    predictor: str = Field(
        description="Clinical variable or biomarker used as predictor."
    )
    predictor_variable: str = Field(
        default="Not reported",
        description=(
            "Explicit predictor variable label for counterfactual extraction mode."
        ),
    )
    outcome_definition: str = Field(
        description=(
            "How the primary outcome is defined, e.g. '28-day all-cause "
            "mortality'."
        )
    )
    timing: str = Field(
        description=(
            "When the predictor was measured relative to admission/diagnosis. "
            "Write 'Not reported' if not stated."
        )
    )
    statistical_method: str = Field(
        description=(
            "Statistical method used, e.g. 'logistic regression', "
            "'ROC analysis'. Write 'Not reported' if not stated."
        )
    )
    effect_size: str = Field(
        description=(
            "Reported effect size with value, e.g. 'OR 1.2', 'HR 2.4', "
            "'AUC 0.78'. Write 'Not reported' if not stated."
        )
    )
    performance_metrics: str = Field(
        description=(
            "Sensitivity, specificity, AUC, p-value, or 95% CI. "
            "Write 'Not reported' if not stated."
        )
    )
    notes: str = Field(
        default="Not reported",
        description="Any additional relevant notes from the study.",
    )
    source: SourceAnchor = Field(
        description="Source anchor linking the extraction to the document."
    )


# ---------------------------------------------------------------------------
# Use Case 2 – Sepsis Phenotype Extraction
# ---------------------------------------------------------------------------

class PhenotypeCluster(BaseModel):
    """Description of a single patient phenotype/cluster."""

    cluster_id: str = Field(
        description="Cluster identifier, e.g. 'A', 'B', '1', '2'."
    )
    cluster_name: str = Field(
        default="Not reported",
        description="Human-readable cluster name if provided in paper.",
    )
    cluster_size: str = Field(
        description=(
            "Cluster size (N or %). Write 'Not reported' if not stated."
        )
    )
    key_features: str = Field(
        description="Key clinical or laboratory features of this cluster."
    )
    clinical_description: str = Field(
        description="Clinical interpretation of this cluster."
    )
    outcome: str = Field(
        description=(
            "Reported outcome for this cluster (e.g. ICU mortality). "
            "Write 'Not reported' if not stated."
        )
    )
    outcomes: str = Field(
        default="Not reported",
        description=(
            "Alternate plural outcomes field for nested phenotype extraction mode."
        ),
    )
    notes: str = Field(
        default="Not reported",
        description="Additional notes about this cluster.",
    )


class UseCase2Evidence(BaseModel):
    """
    Structured evidence for sepsis phenotype extraction.

    Captures the clustering approach, variables used, and per-cluster
    characterisation so phenotypes can be compared across studies.
    """

    study_name: str = Field(
        description="Author surname(s) and year."
    )
    country: str = Field(
        description="Country of the study. Write 'Not reported' if not stated."
    )
    setting: str = Field(
        description=(
            "Clinical setting (ICU, ED, etc.). Write 'Not reported' if not stated."
        )
    )
    sample_size: str = Field(
        description=(
            "Total number of patients. Write 'Not reported' if not stated."
        )
    )
    sepsis_definition: str = Field(
        description=(
            "Sepsis definition used, e.g. 'Sepsis-3'. "
            "Write 'Not reported' if not stated."
        )
    )
    clustering_method: str = Field(
        description=(
            "Method used for phenotype identification, e.g. 'k-means', "
            "'latent class analysis'."
        )
    )
    num_clusters: str = Field(
        description="Number of clusters/phenotypes identified."
    )
    variables_used: List[str] | str = Field(
        description=(
            "Variables used for clustering. Write 'Not reported' if not stated."
        )
    )
    assignment_feasibility: str = Field(
        description=(
            "Whether phenotype assignment rules are reproducible: "
            "'Assignable', 'Not assignable', or 'Insufficient detail'."
        )
    )
    assignment_notes: str = Field(
        description=(
            "Why assignment is (not) possible. Write 'Not reported' if not stated."
        )
    )
    assignment_is_reproducible: bool = Field(
        default=False,
        description="Boolean reproducibility flag for phenotype assignment.",
    )
    reproducibility_notes: str = Field(
        default="Not reported",
        description="Rationale for reproducibility determination.",
    )
    phenotypes: List[PhenotypeCluster] = Field(
        description="List of identified phenotypes/clusters."
    )
    source: SourceAnchor = Field(
        description="Source anchor linking the extraction to the document."
    )


# ---------------------------------------------------------------------------
# Use Case 3 – Biomarker Selection for Risk Stratification
# ---------------------------------------------------------------------------

class UseCase3Evidence(BaseModel):
    """
    Structured evidence for biomarker/score selection for risk stratification.

    Enables cross-study comparison of predictive performance to support
    evidence-based selection of stratification variables.
    """

    study_name: str = Field(
        description="Author surname(s) and year."
    )
    biomarker_or_score: str = Field(
        description="Name of the biomarker or clinical score."
    )
    biomarker_name: str = Field(
        default="Not reported",
        description="Canonical biomarker name for ranking/comparison mode.",
    )
    biomarker_type: str = Field(
        description="Category: 'Biomarker' or 'Clinical Score'."
    )
    population: str = Field(
        description="Patient population."
    )
    cohort_setting: str = Field(
        default="Not reported",
        description="Care setting (e.g., ICU/ED/ward) for direct comparison.",
    )
    sample_size: str = Field(
        description=(
            "Total number of patients. Write 'Not reported' if not stated."
        )
    )
    outcome: str = Field(
        description="Outcome measured, e.g. '28-day mortality'."
    )
    effect_size: str = Field(
        description=(
            "Effect size value, e.g. 'OR 1.5', 'HR 2.1'. "
            "Write 'Not reported' if not stated."
        )
    )
    auc: str = Field(
        description=(
            "Area under the curve (AUC/AUROC). "
            "Write 'Not reported' if not stated."
        )
    )
    auroc: Optional[float] = Field(
        default=None,
        description="Numeric AUROC value for sortable biomarker comparisons.",
    )
    confidence_interval: str = Field(
        description=(
            "95% confidence interval. Write 'Not reported' if not stated."
        )
    )
    adjustment: str = Field(
        description=(
            "Covariate adjustment or model details (e.g., adjusted for age/sex). "
            "Write 'Not reported' if not stated."
        )
    )
    statistical_method: str = Field(
        description="Statistical method used."
    )
    validation_method: str = Field(
        description=(
            "Validation approach (e.g. 'internal', 'external', 'bootstrap'). "
            "Write 'Not reported' if not stated."
        )
    )
    relevance_to_target_population: str = Field(
        description=(
            "Notes on relevance to the target population. "
            "Write 'Not reported' if not stated."
        )
    )
    cohort_characteristics: str = Field(
        description=(
            "Key cohort characteristics. Write 'Not reported' if not stated."
        )
    )
    notes: str = Field(
        default="Not reported",
        description="Additional notes.",
    )
    source: SourceAnchor = Field(
        description="Source anchor linking the extraction to the document."
    )
