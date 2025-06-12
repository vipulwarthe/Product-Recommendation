import pandas as pd
import pickle

# Load model, data, and encoder
with open('model.pkl', 'rb') as f:
    model, pivot_table, product_id_encoder = pickle.load(f)

full_data = pd.read_csv('amazon.csv')

def get_recommendations_with_details(product_id=0, n_recommendations=5):
    original_product_id = product_id_encoder.inverse_transform([product_id])[0]

    product_index = pivot_table.index.get_loc(product_id)

    distances, indices = model.kneighbors(
        pivot_table.iloc[[product_index]], n_neighbors=n_recommendations + 1)

    recommendations = []

    for j in range(1, len(indices[0])):
        try:
            recommended_index = indices[0][j]
            recommended_encoded_id = pivot_table.index[recommended_index]
            recommended_original_id = product_id_encoder.inverse_transform([recommended_encoded_id])[0]

            recommended_product_details = full_data[full_data['product_id'] == recommended_original_id].iloc[0]

            recommendation_info = {
                'product_id': recommended_original_id,
                'product_name': recommended_product_details['product_name'],
                'category': recommended_product_details['category'],
                'rating': recommended_product_details['rating'],
                'rating_count': recommended_product_details['rating_count'],
                'img_link': recommended_product_details['img_link'],
                'product_link': recommended_product_details['product_link'],
                'distance': distances[0][j]
            }

            recommendations.append(recommendation_info)

        except:
            continue

    return recommendations








