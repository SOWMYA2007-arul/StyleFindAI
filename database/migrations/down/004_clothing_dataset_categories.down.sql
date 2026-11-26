-- Rollback of 004. Removes the added categories only if no product uses them (products are never deleted).
DELETE FROM categories c
 WHERE c.slug IN ('outerwear', 'pants', 'shorts', 'skirts', 'footwear', 'hats')
   AND NOT EXISTS (SELECT 1 FROM products p WHERE p.category_id = c.id);
