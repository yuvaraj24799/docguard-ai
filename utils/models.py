from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, Text, Enum, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import uuid, enum, os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"

class InferenceMode(str, enum.Enum):
    BATCH = "batch"
    REALTIME = "realtime"

class ReviewAction(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CORRECTED = "corrected"


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    document_category = Column(String)
    raw_text = Column(Text)
    structured_output = Column(JSON)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    inference_mode = Column(Enum(InferenceMode), default=InferenceMode.REALTIME)
    processing_latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    evaluations = relationship("Evaluation", back_populates="document")
    review_items = relationship("ReviewQueueItem", back_populates="document")
    audit_logs = relationship("AuditLog", back_populates="document")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    correctness_score = Column(Float)
    faithfulness_score = Column(Float)
    completeness_score = Column(Float)
    hallucination_risk_score = Column(Float)
    overall_score = Column(Float)
    judge_reasoning = Column(JSON)
    golden_set_comparison = Column(JSON)
    sources_cited = Column(JSON)
    confidence = Column(Float)
    flagged_for_review = Column(Boolean, default=False)
    flag_reason = Column(String)
    baseline_score = Column(Float)
    drift_detected = Column(Boolean, default=False)
    drift_magnitude = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="evaluations")


class ReviewQueueItem(Base):
    __tablename__ = "review_queue"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    evaluation_id = Column(String, ForeignKey("evaluations.id"))
    original_output = Column(JSON)
    confidence_score = Column(Float)
    flag_reason = Column(String)
    status = Column(String, default="pending")
    reviewer_action = Column(Enum(ReviewAction))
    reviewer_correction = Column(JSON)
    reviewer_notes = Column(Text)
    reviewed_at = Column(DateTime)
    added_to_golden_set = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="review_items")


class GoldenSet(Base):
    __tablename__ = "golden_sets"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_category = Column(String, nullable=False)
    input_text = Column(Text, nullable=False)
    expected_output = Column(JSON, nullable=False)
    source = Column(String)
    quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelDriftAlert(Base):
    __tablename__ = "drift_alerts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_category = Column(String)
    metric_name = Column(String)
    baseline_value = Column(Float)
    current_value = Column(Float)
    drift_magnitude = Column(Float)
    alert_threshold = Column(Float)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    action = Column(String, nullable=False)
    actor = Column(String, default="system")
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="audit_logs")


def get_engine():
    return create_engine(os.getenv("DATABASE_URL", "sqlite:///./docguard.db"))

def get_session():
    return sessionmaker(bind=get_engine())()

def create_tables():
    Base.metadata.create_all(get_engine())
    print("tables ok")

if __name__ == "__main__":
    create_tables()
