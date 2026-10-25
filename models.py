import os
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import CheckConstraint, UniqueConstraint

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

    @property
    def is_buyer(self):
        return self.role == "buyer"

    @property
    def is_seller(self):
        return self.role == "seller"

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

    def __repr__(self):
        return f"<Inquiry #{self.id} Listing:{self.listing_id} Buyer:{self.buyer_id}->Seller:{self.seller_id}>"
