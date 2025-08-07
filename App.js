import React, { useEffect, useState } from 'react';
import { SafeAreaView, Text, TouchableOpacity, FlatList } from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';

export default function App() {
  const { model, error } = useTensorflowModel(
    require('./src/assets/output_predection.tflite'),
  );
  const [order, setOrder] = useState({ burger: 0, pizza: 0, coffee: 0 });
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    if (error) console.error('❌ Model load error:', error);
  }, [error]);

  useEffect(() => {
    if (model && (order.burger || order.pizza || order.coffee)) {
      runPrediction();
    } else {
      setRecommendations([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [order, model]);

  const runPrediction = () => {
    // Input as Float32Array (e.g., Burger=1, Pizza=0, Coffee=0)
    const input = new Float32Array([order.burger, order.pizza, order.coffee]);

    const output = model.runSync([input]);
    console.log('📊 Predictions:', output);

    const productNames = ['Fries', 'Coke', 'Muffin', 'Cookie'];
    const mapped = productNames.map((name, idx) => ({
      label: name,
      confidence: output[0][idx] || 0,
    }));

    setRecommendations(mapped.sort((a, b) => b.confidence - a.confidence));
  };

  const toggleItem = item => {
    setOrder(prev => ({
      ...prev,
      [item]: prev[item] ? 0 : 1,
    }));
  };

  const menuItems = [
    { id: 'burger', label: 'Burger' },
    { id: 'pizza', label: 'Pizza' },
    { id: 'coffee', label: 'Coffee' },
  ];

  return (
    <SafeAreaView style={{ flex: 1, padding: 20 }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold' }}>POS Order Screen</Text>

      <Text style={{ marginTop: 20, fontSize: 18 }}>Select Items:</Text>
      {menuItems.map(item => (
        <TouchableOpacity
          key={item.id}
          onPress={() => toggleItem(item.id)}
          style={{
            padding: 12,
            backgroundColor: order[item.id] ? 'green' : 'gray',
            marginVertical: 5,
            borderRadius: 8,
          }}
        >
          <Text style={{ color: 'white', fontSize: 16 }}>{item.label}</Text>
        </TouchableOpacity>
      ))}

      <Text style={{ marginTop: 20, fontSize: 18 }}>AI Recommendations:</Text>
      <FlatList
        data={recommendations}
        keyExtractor={item => item.label}
        renderItem={({ item }) => (
          <Text style={{ fontSize: 16 }}>
            {item.label} — {(item.confidence * 100).toFixed(1)}%
          </Text>
        )}
      />
    </SafeAreaView>
  );
}
