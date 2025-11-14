import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

df = pd.read_csv('Apartment_Market_Prices new.csv')

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nBasic statistics:")
print(df.describe())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Apartment Market Price Analysis', fontsize=16, fontweight='bold')

yearly_data = df.groupby('Year').agg({
    'Tract Median Apartment Contract Rent per Square Foot': 'mean',
    'Tract Median Apartment Contract Rent per Unit': 'mean'
}).reset_index()

ax1 = axes[0, 0]
ax1.plot(yearly_data['Year'], yearly_data['Tract Median Apartment Contract Rent per Square Foot'], 
         marker='o', linewidth=2, markersize=6, color='#2E86AB')
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Rent per Sq Ft ($)', fontweight='bold')
ax1.set_title('Average Rent per Square Foot Over Time')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(yearly_data['Year'], yearly_data['Tract Median Apartment Contract Rent per Unit'], 
         marker='s', linewidth=2, markersize=6, color='#A23B72')
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Rent per Unit ($)', fontweight='bold')
ax2.set_title('Average Rent per Unit Over Time')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
cost_category_counts = df['Cost Category'].value_counts()
colors = ['#F18F01', '#C73E1D', '#6A994E']
ax3.bar(cost_category_counts.index, cost_category_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax3.set_xlabel('Cost Category', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Distribution of Cost Categories')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = axes[1, 1]
top_areas = df.groupby('Community Reporting Area Name').agg({
    'Tract Median Apartment Contract Rent per Unit': 'mean'
}).nlargest(10, 'Tract Median Apartment Contract Rent per Unit')
ax4.barh(top_areas.index, top_areas['Tract Median Apartment Contract Rent per Unit'], 
         color='#4A8FE7', alpha=0.8, edgecolor='black')
ax4.set_xlabel('Average Rent per Unit ($)', fontweight='bold')
ax4.set_ylabel('Area', fontweight='bold')
ax4.set_title('Top 10 Most Expensive Areas by Average Rent')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/apartment_analysis_overview.png', dpi=300, bbox_inches='tight')
print("\nSaved: apartment_analysis_overview.png")
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Detailed Apartment Market Analysis', fontsize=16, fontweight='bold')

ax5 = axes2[0, 0]
rent_data = df[df['Tract Median Apartment Contract Rent per Square Foot'] > 0]['Tract Median Apartment Contract Rent per Square Foot']
ax5.hist(rent_data, bins=50, color='#06A77D', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Rent per Square Foot ($)', fontweight='bold')
ax5.set_ylabel('Frequency', fontweight='bold')
ax5.set_title('Distribution of Rent per Square Foot')
ax5.grid(True, alpha=0.3, axis='y')

ax6 = axes2[0, 1]
properties_data = df[df['PROPERTIES'] > 0]['PROPERTIES']
ax6.hist(properties_data, bins=50, color='#D81159', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Number of Properties', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title('Distribution of Properties per Tract')
ax6.grid(True, alpha=0.3, axis='y')

ax7 = axes2[1, 0]
area_counts = df['Community Reporting Area Name'].value_counts().head(15)
ax7.barh(range(len(area_counts)), area_counts.values, color='#8338EC', alpha=0.8, edgecolor='black')
ax7.set_yticks(range(len(area_counts)))
ax7.set_yticklabels(area_counts.index, fontsize=9)
ax7.set_xlabel('Number of Records', fontweight='bold')
ax7.set_ylabel('Community Area', fontweight='bold')
ax7.set_title('Top 15 Areas by Number of Records')
ax7.grid(True, alpha=0.3, axis='x')

ax8 = axes2[1, 1]
change_category = df['Year over Year Change in Rent Category'].value_counts()
colors_change = ['#FF6B6B', '#4ECDC4', '#FFE66D']
wedges, texts, autotexts = ax8.pie(change_category.values, labels=change_category.index, 
                                     autopct='%1.1f%%', startangle=90, colors=colors_change)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax8.set_title('Year-over-Year Rent Change Category Distribution')

plt.tight_layout()
plt.savefig('visualizations/apartment_analysis_detailed.png', dpi=300, bbox_inches='tight')
print("Saved: apartment_analysis_detailed.png")
plt.show()

fig3, ax = plt.subplots(figsize=(14, 8))
scatter_data = df[(df['Tract Median Apartment Contract Rent per Unit'] > 0) & 
                  (df['PROPERTIES'] > 0)].copy()
scatter_data = scatter_data.sample(min(1000, len(scatter_data)))
scatter = ax.scatter(scatter_data['PROPERTIES'], 
                     scatter_data['Tract Median Apartment Contract Rent per Unit'],
                     c=scatter_data['Year'], cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Properties', fontweight='bold', fontsize=12)
ax.set_ylabel('Rent per Unit ($)', fontweight='bold', fontsize=12)
ax.set_title('Relationship Between Number of Properties and Rent per Unit', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Year', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/properties_vs_rent_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: properties_vs_rent_scatter.png")
plt.show()

print("\n" + "="*50)
print("Visualization complete! Generated 3 figures:")
print("1. apartment_analysis_overview.png")
print("2. apartment_analysis_detailed.png")
print("3. properties_vs_rent_scatter.png")
print("="*50)
