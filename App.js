import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  Text,
  TouchableOpacity,
  TextInput,
  FlatList,
} from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';

const menuItems = [
  'Burger Meal',
  'Burger Combo',
  'Pizza Large',
  'Pizza Medium',
  'Coffee Latte',
  'Coffee Black',
  'Garlic Bread',
  'Salad',
];

export default function App() {
  // Load AI models
  const { model: comboModel } = useTensorflowModel(
    require('./src/assets/pos_combo_model.tflite'),
  );
  const { model: kitchenModel } = useTensorflowModel(
    require('./src/assets/kitchen_load_model.tflite'),
  );

  // Order & Recommendation State
  const [order, setOrder] = useState({ burger: 0, pizza: 0, coffee: 0 });
  const [recommendations, setRecommendations] = useState([]);

  // Autocomplete State
  const [typed, setTyped] = useState('');
  const [suggestions, setSuggestions] = useState([]);

  // Kitchen Prediction
  const [kitchenLoad, setKitchenLoad] = useState(null);

  const toggleItem = item => {
    setOrder(prev => ({ ...prev, [item]: prev[item] ? 0 : 1 }));
  };

  // Run combo recommendation model
  useEffect(() => {
    if (!comboModel) return;
    const input = new Float32Array([order.burger, order.pizza, order.coffee]);
    const output = comboModel.runSync([input]);
    const names = [
      'Fries',
      'Coke',
      'Muffin',
      'Cookie',
      'Garlic Bread',
      'Salad',
    ];
    const mapped = names.map((n, i) => ({ name: n, score: output[0][i] || 0 }));
    setRecommendations(mapped.filter(m => m.score > 0.5)); // only >50%
  }, [order, comboModel]);

  // Auto-complete logic
  useEffect(() => {
    if (!typed) return setSuggestions([]);
    const lower = typed.toLowerCase();
    setSuggestions(
      menuItems.filter(m => m.toLowerCase().startsWith(lower)).slice(0, 3),
    );
  }, [typed]);

  // Run kitchen load prediction for fries (demo)
  const predictKitchenLoad = () => {
    if (!kitchenModel) return;
    const now = new Date();
    const hour = now.getHours();
    const day = now.getDay();
    const input = new Float32Array([hour, day]);
    const output = kitchenModel.runSync([input]);
    setKitchenLoad(Math.round(output[0][0]));
  };

  return (
    <SafeAreaView style={{ flex: 1, padding: 20 }}>
      {/* POS Order Screen */}
      <Text style={{ fontSize: 24, fontWeight: 'bold' }}>POS Order</Text>
      {['burger', 'pizza', 'coffee'].map(item => (
        <TouchableOpacity
          key={item}
          onPress={() => toggleItem(item)}
          style={{
            padding: 12,
            backgroundColor: order[item] ? 'green' : 'gray',
            marginVertical: 5,
            borderRadius: 8,
          }}
        >
          <Text style={{ color: 'white', fontSize: 16 }}>
            {item.toUpperCase()}
          </Text>
        </TouchableOpacity>
      ))}

      <Text style={{ marginTop: 15, fontSize: 18 }}>
        AI Combo Recommendations:
      </Text>
      <FlatList
        style={{ flex: 1, maxHeight: 200 }}
        contentContainerStyle={{ height: 200 }}
        data={recommendations}
        keyExtractor={item => item.name}
        renderItem={({ item }) => (
          <Text style={{ fontSize: 16 }}>
            {item.name} — {(item.score * 100).toFixed(1)}%
          </Text>
        )}
      />

      {/* Autocomplete Input */}
      <Text style={{ marginTop: 20, fontSize: 18 }}>Search Menu Item:</Text>
      <TextInput
        style={{ borderWidth: 1, padding: 8, marginVertical: 5 }}
        placeholder="Type item..."
        value={typed}
        onChangeText={setTyped}
      />
      {suggestions.map(s => (
        <Text key={s} style={{ fontSize: 16 }}>
          {s}
        </Text>
      ))}

      {/* Kitchen Load Prediction */}
      <TouchableOpacity
        onPress={predictKitchenLoad}
        style={{
          marginTop: 20,
          backgroundColor: 'orange',
          padding: 12,
          borderRadius: 8,
        }}
      >
        <Text style={{ color: 'white', fontSize: 16 }}>
          Predict Kitchen Load
        </Text>
      </TouchableOpacity>
      {kitchenLoad !== null && (
        <Text style={{ fontSize: 16, marginTop: 10 }}>
          Expected Fries Demand: {kitchenLoad}
        </Text>
      )}
    </SafeAreaView>
  );
}
