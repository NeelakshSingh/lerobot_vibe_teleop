import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  SafeAreaView,
  Platform,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { DeviceMotion } from "expo-sensors";

// ── Helpers ──────────────────────────────────────────────────────────────

/** Convert Expo DeviceMotion rotation (Euler) to quaternion [w,x,y,z]. */
function eulerToQuat(
  alpha: number, // Z-axis rotation (yaw)
  beta: number, // X-axis rotation (pitch)
  gamma: number // Y-axis rotation (roll)
): [number, number, number, number] {
  const ha = alpha / 2;
  const hb = beta / 2;
  const hg = gamma / 2;
  const ca = Math.cos(ha);
  const sa = Math.sin(ha);
  const cb = Math.cos(hb);
  const sb = Math.sin(hb);
  const cg = Math.cos(hg);
  const sg = Math.sin(hg);
  // ZXY intrinsic order (common for device orientation)
  const w = ca * cb * cg - sa * sb * sg;
  const x = ca * sb * cg - sa * cb * sg;
  const y = ca * sb * sg + sa * cb * cg;
  const z = ca * cb * sg + sa * sb * cg;
  return [w, x, y, z];
}

// ── App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [serverIp, setServerIp] = useState("10.1.3.191");
  const [connected, setConnected] = useState(false);
  const [grasp, setGrasp] = useState(false);
  const [eePos, setEePos] = useState<number[] | null>(null);
  const [sensorActive, setSensorActive] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const graspRef = useRef(false);

  // Keep graspRef in sync so the sensor callback always has the latest.
  useEffect(() => {
    graspRef.current = grasp;
  }, [grasp]);

  // ── WebSocket ──────────────────────────────────────────────────────

  const connect = useCallback(() => {
    if (wsRef.current) return;
    const url = `ws://${serverIp}:8765`;
    console.log(`Connecting to ${url} …`);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log("WS connected");
      setConnected(true);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "state" && msg.ee_pos) {
          setEePos(msg.ee_pos);
        }
      } catch {}
    };

    ws.onerror = (e) => console.log("WS error", e);

    ws.onclose = () => {
      console.log("WS closed");
      setConnected(false);
      wsRef.current = null;
    };

    wsRef.current = ws;
  }, [serverIp]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  // ── DeviceMotion ───────────────────────────────────────────────────

  useEffect(() => {
    if (!connected) {
      setSensorActive(false);
      return;
    }

    DeviceMotion.setUpdateInterval(1000 / 30); // 30 Hz
    const sub = DeviceMotion.addListener((data) => {
      const rot = data.rotation;
      if (!rot || wsRef.current?.readyState !== WebSocket.OPEN) return;

      const quat = eulerToQuat(rot.alpha, rot.beta, rot.gamma);

      const payload = JSON.stringify({
        type: "imu",
        quaternion: quat,
        grasp: graspRef.current,
        timestamp: Date.now(),
      });
      wsRef.current.send(payload);
      setSensorActive(true);
    });

    return () => sub.remove();
  }, [connected]);

  // ── Recalibrate ────────────────────────────────────────────────────

  const recalibrate = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "recalibrate" }));
    }
  }, []);

  // ── UI ─────────────────────────────────────────────────────────────

  const fmtPos = (pos: number[] | null) =>
    pos ? pos.map((v) => v.toFixed(3)).join(", ") : "—";

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />

      {/* ── Header ─────────────────────────────────────────────── */}
      <View style={styles.header}>
        <Text style={styles.title}>SO-101 Teleop</Text>
        <View style={styles.statusRow}>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: connected ? "#4caf50" : "#f44336" },
            ]}
          />
          <Text style={styles.statusText}>
            {connected
              ? sensorActive
                ? "Streaming"
                : "Connected"
              : "Disconnected"}
          </Text>
        </View>
      </View>

      {/* ── Server IP + Connect ────────────────────────────────── */}
      <View style={styles.connectRow}>
        <TextInput
          style={styles.ipInput}
          value={serverIp}
          onChangeText={setServerIp}
          placeholder="Server IP"
          placeholderTextColor="#888"
          keyboardType="numeric"
          autoCorrect={false}
          autoCapitalize="none"
        />
        <TouchableOpacity
          style={[
            styles.connectBtn,
            { backgroundColor: connected ? "#f44336" : "#4caf50" },
          ]}
          onPress={connected ? disconnect : connect}
        >
          <Text style={styles.btnText}>
            {connected ? "Disconnect" : "Connect"}
          </Text>
        </TouchableOpacity>
      </View>

      {/* ── EE Position readout ────────────────────────────────── */}
      <View style={styles.infoBox}>
        <Text style={styles.infoLabel}>EE Position</Text>
        <Text style={styles.infoValue}>{fmtPos(eePos)}</Text>
      </View>

      {/* ── Recalibrate ────────────────────────────────────────── */}
      <TouchableOpacity
        style={styles.recalibrateBtn}
        onPress={recalibrate}
        disabled={!connected}
      >
        <Text style={styles.btnText}>Recalibrate</Text>
      </TouchableOpacity>

      {/* ── GRASP button (fills bottom half) ───────────────────── */}
      <TouchableOpacity
        style={[
          styles.graspBtn,
          { backgroundColor: grasp ? "#f44336" : "#2196f3" },
        ]}
        onPress={() => setGrasp((g) => !g)}
        activeOpacity={0.7}
        disabled={!connected}
      >
        <Text style={styles.graspText}>{grasp ? "RELEASE" : "GRASP"}</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

// ── Styles ───────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#1a1a2e",
    paddingTop: Platform.OS === "android" ? 30 : 0,
  },
  header: {
    alignItems: "center",
    paddingVertical: 12,
  },
  title: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "700",
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 6,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  statusText: {
    color: "#ccc",
    fontSize: 14,
  },
  connectRow: {
    flexDirection: "row",
    paddingHorizontal: 16,
    marginTop: 8,
    gap: 8,
  },
  ipInput: {
    flex: 1,
    backgroundColor: "#16213e",
    color: "#fff",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 16,
    borderWidth: 1,
    borderColor: "#333",
  },
  connectBtn: {
    borderRadius: 8,
    paddingHorizontal: 20,
    justifyContent: "center",
  },
  btnText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  infoBox: {
    marginHorizontal: 16,
    marginTop: 14,
    backgroundColor: "#16213e",
    borderRadius: 8,
    padding: 12,
  },
  infoLabel: {
    color: "#888",
    fontSize: 12,
    marginBottom: 4,
  },
  infoValue: {
    color: "#fff",
    fontSize: 16,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
  },
  recalibrateBtn: {
    marginHorizontal: 16,
    marginTop: 12,
    backgroundColor: "#e65100",
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: "center",
  },
  graspBtn: {
    flex: 1,
    marginHorizontal: 16,
    marginTop: 14,
    marginBottom: 16,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  graspText: {
    color: "#fff",
    fontSize: 42,
    fontWeight: "800",
    letterSpacing: 4,
  },
});
