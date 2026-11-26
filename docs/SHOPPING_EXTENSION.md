# Optional shopping extension - design only (not implemented)

Checked first: v2 has **no** customer accounts, cart, wishlist, orders, payments or seller accounts. As the brief
requires, these are kept separate from visual search and are only designed here.

## Separation
New tables in a new migration (`004_shop.sql`), new `backend/src/shop/*` routes, no changes to the AI service.
Visual search stays anonymous and works without login.

## Data model (sketch)
| table | key columns |
|---|---|
| customers | id, email UNIQUE, password_hash (bcrypt), name, created_at, token_version |
| addresses | id, customer_id FK, line1, city, state, pincode, phone |
| carts / cart_items | cart_id FK, product_id FK, quantity CHECK > 0, UNIQUE(cart_id, product_id) |
| wishlists | customer_id FK, product_id FK, PRIMARY KEY(customer_id, product_id) |
| orders | id, customer_id FK, status (pending/paid/shipped/delivered/cancelled), total NUMERIC, currency, created_at |
| order_items | order_id FK, product_id FK, unit_price NUMERIC (copied at order time), quantity |
| payments | order_id FK, provider, provider_payment_id UNIQUE, status, amount - **no card data ever stored** |
| sellers / seller_products (later) | seller_id, product_id, approval_status (pending/approved/rejected), reviewed_by admin |

Products currently have optional prices; stock (`inventory` table) would be needed before checkout.

## Access control
Roles: `customer`, `seller`, `admin` in separate JWT audiences/cookies; sellers may only modify their own products
(row-level check `seller_products.seller_id = current seller`), admin approves listings. Same cookie + CSRF scheme as admin.

## Payments
Use a provider's hosted checkout (e.g. Razorpay or Stripe Checkout): the browser is redirected to the provider, the
backend verifies the signed webhook and marks the order paid. Raw card numbers never reach StyleFind. Requires a
merchant account (a paid service) - needs your approval before implementation.
