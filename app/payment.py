import stripe
from fastapi import HTTPException
import os

# Set up the Stripe API key
stripe.api_key = os.getenv(
    "STRIPE_SECRET_KEY", "sk_test_51QFSSY05XemVskwXbeESlpgekSs24cK93Hg0hN7JHC1RYf8JM5x0zcanl2w2enIC3LP4wnABl61QcgVsIRnLmFF100IENR1uC9"
)


async def create_payment_intent(amount: int, user_id: str):
    """
    Creates a Stripe PaymentIntent for the specified amount in SGD.
    """
    try:
        # Create a PaymentIntent with the specified amount, currency, and metadata
        payment_intent = stripe.PaymentIntent.create(
            amount=amount,  # Amount in cents for SGD
            currency="sgd",
            metadata={"user_id": user_id},  # Include user ID in metadata
            # Enable automatic payment methods
            automatic_payment_methods={"enabled": True}
        )
        # Log the PaymentIntent ID
        print(f"Created PaymentIntent: {payment_intent.id}")
        return {"client_secret": payment_intent.client_secret}
    except Exception as e:
        print(f"Error creating PaymentIntent: {str(e)}")  # Log the error
        raise HTTPException(status_code=400, detail=str(e))


async def create_subscription(amount: int, user_id: str):
    """
    Creates a Stripe Subscription for the specified amount in SGD.
    """
    try:
        # Create a product
        product = stripe.Product.create(name="Monthly Subscription")

        # Create a price for the product
        price = stripe.Price.create(
            product=product.id,
            unit_amount=amount,
            currency="sgd",
            recurring={"interval": "month"}
        )

        # Create a customer
        customer = stripe.Customer.create(
            metadata={"user_id": user_id})  # Include user ID in metadata

        # Create a subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": price.id}],
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"]
        )

        return {"client_secret": subscription.latest_invoice.payment_intent.client_secret}
    except Exception as e:
        print(f"Error creating Subscription: {str(e)}")  # Log the error
        raise HTTPException(status_code=400, detail=str(e))
