# Plot price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['price'] / 100000, bins=100, kde=True)
plt.xlim(0, 500)  # Limit to 0–500 lakhs (₹0–₹5Cr)
plt.title("Price Distribution (Zoomed In)")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()