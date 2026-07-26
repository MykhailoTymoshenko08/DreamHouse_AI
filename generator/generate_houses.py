import pandas as pd
import numpy as np
import os

def generate_dream_houses(n_samples=5000):
    print(f"Починаємо генерацію {n_samples} будинків...")
    np.random.seed(42) # Щоб дані завжди генерувалися однаково
    
    # Square 50-100 sq m
    area_sqm = np.random.uniform(50, 500, n_samples)
    
    # Rooms
    rooms = np.maximum(1, np.round(area_sqm / np.random.uniform(30, 50, n_samples))).astype(int)
    
    # Floor
    floors = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.4, 0.1])
    
    # Distance from the center of city
    distance_km = np.random.uniform(1, 100, n_samples)
    
    # Materials: 0 - Tree (eco, but expensive), 1 - Gasblock (cheap), 2 - Bricks (classic, middle price)
    materials = np.random.choice(['Wood', 'Aerated_block', 'Brick'], n_samples, p=[0.2, 0.5, 0.3])
    
    # Swimming bool (if area > 150 so chance of pool greater)
    has_pool = np.where(area_sqm > 150, np.random.choice([0, 1], n_samples, p=[0.6, 0.4]), 
                                        np.random.choice([0, 1], n_samples, p=[0.95, 0.05]))
    
    # Landscape design
    has_landscape = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # 2. Logic of price calculation
    # Basic price of the box
    base_price_per_sqm = np.zeros(n_samples)
    base_price_per_sqm[materials == 'Aerated_block'] = 400  # The cheapest
    base_price_per_sqm[materials == 'Brick'] = 600          # Middle
    base_price_per_sqm[materials == 'Wood'] = 800           # Expincive

    price = area_sqm * base_price_per_sqm
    
    # Additional expenses
    price += has_pool * 25000       # Pool $25k
    price += has_landscape * 10000  # Landscape $10k
    price += (floors > 1) * 15000   # Stairs + perekritya
    
    # Each km make price on $200 less
    price -= distance_km * 200
    
    noise = np.random.normal(0, 15000, n_samples)
    final_price = np.maximum(10000, price + noise) # Not less price than $10k

    # 3. Locic of energoefe
    # A - Economy, B - Standart, C - Expensive
    energy_score = (area_sqm * 0.5) + (has_pool * 50) + (distance_km * 0.1)
    energy_score[materials == 'Wood'] -= 30
    energy_score[materials == 'Brick'] += 20
    
    energy_class = np.where(energy_score < 100, 'A', 
                   np.where(energy_score < 200, 'B', 'C'))

    # 4. Packing to the dataframe
    df = pd.DataFrame({
        'area_sqm': np.round(area_sqm, 1),
        'rooms': rooms,
        'floors': floors,
        'distance_city_km': np.round(distance_km, 1),
        'material': materials,
        'has_pool': has_pool,
        'has_landscape': has_landscape,
        'price_usd': np.round(final_price, 0),
        'energy_class': energy_class
    })

    os.makedirs('../data/raw', exist_ok=True)
    
    file_path = '../data/raw/houses_dataset.csv'
    df.to_csv(file_path, index=False)
    print(f"Done! Were generated {n_samples} notes.")
    print(f"File was saveed by: {file_path}")
    
    print("\nFirst 5:")
    print(df.head())

if __name__ == "__main__":
    generate_dream_houses()