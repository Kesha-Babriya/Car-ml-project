import hashlib
import os
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import CheckConstraint, UniqueConstraint
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


def utcnow():
    """Returns current UTC time aware or naive compatible with SQLite."""
    return datetime.now(timezone.utc)


class User(db.Model):
    """User account entity with explicit role separation between buyer and seller."""

    __tablename__ = "users"

    __table_args__ = (
        CheckConstraint(
            "role IN ('buyer', 'seller')",
            name="chk_user_role"
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), nullable=False, default="buyer")
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    # Relationships
    listings = db.relationship(
        "CarListing",
        backref="seller",
        lazy=True,
        cascade="all, delete-orphan",
        foreign_keys="CarListing.seller_id"
    )
    favourites = db.relationship(
        "Favourite",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )
    predictions = db.relationship(
        "PredictionHistory",
        backref="user",
        lazy=True
    )
    inquiry_messages = db.relationship(
        "InquiryMessage",
        backref="sender",
        lazy=True,
        cascade="all, delete-orphan",
        foreign_keys="InquiryMessage.sender_id"
    )

    @property
    def is_buyer(self):
        return self.role == "buyer"

    @property
    def is_seller(self):
        return self.role == "seller"

    def set_password(self, password):
        """Hashes and stores the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verifies the password against the stored password hash."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    @property
    def auth_fingerprint(self):
        """Short SHA-256 fingerprint of password_hash for session verification."""
        if not self.password_hash:
            return ""
        return hashlib.sha256(self.password_hash.encode("utf-8")).hexdigest()[:16]

    def has_favourited(self, listing_id):
        """Checks if the user has bookmarked a given car listing."""
        return any(f.listing_id == listing_id for f in self.favourites)

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"


class CarListing(db.Model):
    """Seller-published car listing in the marketplace."""

    __tablename__ = "car_listings"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'sold', 'draft')",
            name="chk_listing_status"
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    brand = db.Column(db.String(50), nullable=False)
    model = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    km_driven = db.Column(db.Integer, nullable=False)
    fuel_type = db.Column(db.String(20), nullable=False)
    transmission = db.Column(db.String(20), nullable=False)
    owner = db.Column(db.String(30), nullable=False)
    asking_price = db.Column(db.Float, nullable=False)
    ai_predicted_price = db.Column(db.Float, nullable=True)
    title = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="active", index=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    favourites = db.relationship(
        "Favourite",
        backref="listing",
        lazy=True,
        cascade="all, delete-orphan"
    )
    inquiries = db.relationship(
        "Inquiry",
        backref="listing",
        lazy=True,
        cascade="all, delete-orphan"
    )

    @property
    def formatted_price(self):
        """Formats asking price in Indian Rupees format."""
        try:
            return f"₹ {int(self.asking_price):,}"
        except (ValueError, TypeError):
            return f"₹ {self.asking_price}"

    @property
    def formatted_km(self):
        """Formats mileage with comma separation."""
        try:
            return f"{int(self.km_driven):,} km"
        except (ValueError, TypeError):
            return f"{self.km_driven} km"

    @property
    def is_active(self):
        return self.status == "active"

    @property
    def is_sold(self):
        return self.status == "sold"

    def __repr__(self):
        return f"<CarListing {self.id}: {self.brand} {self.model} ({self.status})>"


class Favourite(db.Model):
    """Buyer bookmarked / saved car listing."""

    __tablename__ = "favourites"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "listing_id",
            name="uq_user_listing_favourite"
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    listing_id = db.Column(
        db.Integer,
        db.ForeignKey("car_listings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def __repr__(self):
        return f"<Favourite User:{self.user_id} Listing:{self.listing_id}>"


class PredictionHistory(db.Model):
    """Valuation log. Supports both authenticated users and anonymous guests (user_id=None)."""

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    brand = db.Column(db.String(50), nullable=False)
    model = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    km_driven = db.Column(db.Integer, nullable=False)
    fuel_type = db.Column(db.String(20), nullable=False)
    transmission = db.Column(db.String(20), nullable=False)
    owner = db.Column(db.String(30), nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def __repr__(self):
        return f"<PredictionHistory {self.brand} {self.model} -> {self.predicted_price}>"


class Inquiry(db.Model):
    """Direct inquiry or price offer from a buyer to a seller regarding a listing."""

    __tablename__ = "inquiries"

    __table_args__ = (
        CheckConstraint(
            "status IN ('unread', 'read', 'replied', 'closed')",
            name="chk_inquiry_status"
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    listing_id = db.Column(
        db.Integer,
        db.ForeignKey("car_listings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    buyer_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    seller_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    message = db.Column(db.Text, nullable=False)
    offer_price = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="unread", index=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    # Relationships
    buyer = db.relationship(
        "User",
        foreign_keys=[buyer_id],
        backref="sent_inquiries"
    )
    seller = db.relationship(
        "User",
        foreign_keys=[seller_id],
        backref="received_inquiries"
    )
    messages = db.relationship(
        "InquiryMessage",
        backref="inquiry",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="InquiryMessage.created_at.asc()"
    )

    def unread_count_for(self, user_id):
        """Returns the number of unread messages sent by the counterpart to this user."""
        if self.messages:
            return sum(1 for m in self.messages if m.sender_id != user_id and not m.is_read)
        # Fallback for legacy enquiries with no separate InquiryMessage rows:
        if user_id == self.seller_id and self.status == "unread":
            return 1
        return 0

    @property
    def conversation_messages(self):
        """Returns all messages in this thread. If no InquiryMessage records exist yet
        (legacy enquiry), synthesizes an initial message so thread view always works."""
        if self.messages:
            return self.messages

        class SyntheticMessage:
            def __init__(self, inq):
                self.id = 0
                self.inquiry_id = inq.id
                self.sender_id = inq.buyer_id
                self.sender = inq.buyer
                self.message = inq.message
                self.is_read = (inq.status in ("read", "replied", "closed"))
                self.created_at = inq.created_at
                self.is_synthetic = True

            def __repr__(self):
                return f"<SyntheticMessage Inq:{self.inquiry_id}>"

        return [SyntheticMessage(self)]

    @property
    def formatted_offer_price(self):
        """Formats optional buyer offer price in Indian Rupees format."""
        if self.offer_price is not None:
            try:
                return f"₹ {int(self.offer_price):,}"
            except (ValueError, TypeError):
                return f"₹ {self.offer_price}"
        return None

    def __repr__(self):
        return f"<Inquiry #{self.id} Listing:{self.listing_id} Buyer:{self.buyer_id}->Seller:{self.seller_id}>"

class PasswordResetToken(db.Model):
    """Stores cryptographically hashed, single-use, time-limited password reset tokens."""

    __tablename__ = "password_reset_tokens"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    token_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    # Relationships
    user = db.relationship(
        "User",
        backref=db.backref("reset_tokens", lazy=True, cascade="all, delete-orphan")
    )

    @classmethod
    def hash_token(cls, raw_token):
        """Hashes the raw token using SHA-256 for secure database storage."""
        if not raw_token:
            return ""
        return hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()

    @property
    def is_expired(self):
        """Returns True if the token has passed its expiration time."""
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            now = datetime.utcnow()
        return now > expires

    @property
    def is_valid(self):
        """Returns True if the token has not been consumed and is not expired."""
        return self.used_at is None and not self.is_expired

    def __repr__(self):
        return f"<PasswordResetToken User:{self.user_id} Exp:{self.expires_at} Used:{bool(self.used_at)}>"

class InquiryMessage(db.Model):
    """An individual message in an inquiry conversation thread between buyer and seller."""

    __tablename__ = "inquiry_messages"

    id = db.Column(db.Integer, primary_key=True)
    inquiry_id = db.Column(
        db.Integer,
        db.ForeignKey("inquiries.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    sender_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def __repr__(self):
        return f"<InquiryMessage #{self.id} Inq:{self.inquiry_id} Sender:{self.sender_id} Read:{self.is_read}>"