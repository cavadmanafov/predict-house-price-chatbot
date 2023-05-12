from typing import Any, Text, Dict, List
import re
import joblib
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import np


with open('assets/location_encoder2.joblib', 'rb') as le:
    location_encoder = joblib.load(le)

with open('assets/xgbmodel3.joblib', 'rb') as f:
    model_xgb = joblib.load(f)


def predict_price(location, size, rooms, floor, model):
    location = float(location)
    size = float(size)
    rooms = float(rooms)
    floor = float(floor)
    X = np.array([[floor, rooms, location, size]])
    predicted_value = model.predict(X)
    return predicted_value


class ActionPredictPrice(Action):

    def name(self) -> Text:
        return "action_predict_price"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            rooms = next(tracker.get_latest_entity_values("rooms"), None)
            location = next(tracker.get_latest_entity_values("location"), None)
            size = next(tracker.get_latest_entity_values("size"), None)
            floor = next(tracker.get_latest_entity_values("floor"), None)


            num_pattern = r'\d+'
            
            rooms_match = re.search(num_pattern, rooms)
            if rooms_match:
                rooms_number = rooms_match.group()

            floor_match = re.search(num_pattern, floor)
            if floor_match:
                floor_number = floor_match.group()

            size_match = re.search(num_pattern, size)
            if size_match:
                size_number = size_match.group()

            if " qəsəbəsi" in location:
                location = location.replace(" qəsəbəsi", "")
            elif " rayonu" in location:
                location = location.replace(" rayonu", "")
            elif " metrosu" in location:
                location = location.replace(" metrosu", "")

            location_encoded = location_encoder.transform([location])[0]

            predicted_price = predict_price(location_encoded, size_number, rooms_number, floor_number, model_xgb)


            dispatcher.utter_message(text=f"Evin təxmin olunan qiyməti: {predicted_price[0]}")

            return []
