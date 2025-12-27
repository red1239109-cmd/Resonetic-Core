# ==============================================================================
# Project: Resonetics
# File:resonetics_os 
# Author: red1239109-cdm
#
# Copyright 2025 red1239109-cdm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import streamlit as st
from resonetics_core import GLOBAL_STATE, Config, PhysicsState, ResoneticAgent

st.set_page_config(layout="wide", page_title=f"Resonetics {Config.VERSION}")

if "AGENT" not in st.session_state:
    st.session_state.AGENT = ResoneticAgent()

agent = st.session_state.AGENT

st.sidebar.title(f"💎 Resonetics {Config.VERSION}")

if GLOBAL_STATE.quarantine:
    st.sidebar.error(f"🚨 QUARANTINE: {GLOBAL_STATE.last_error_code}")
elif GLOBAL_STATE.integrity_degraded:
    st.sidebar.warning("🛡️ DEGRADED")

with st.sidebar.expander("📜 Logs"):
    st.text_area("Recent", "\n".join(GLOBAL_STATE.boot_logs), height=200)

if st.sidebar.button("Start / Stop"):
    with GLOBAL_STATE._lock:
        GLOBAL_STATE.running = not GLOBAL_STATE.running
    st.rerun()

@st.fragment(run_every=Config.PULSE_INTERVAL)
def pulse():
    if GLOBAL_STATE.running:
        agent.tick()
    heat = agent.physics.heat_level()
    phi = round(agent.physics.current_phi, 4)
    st.markdown(f"### 🔥 Heat: {heat} | Φ={phi}")
    st.write(f"Step: {agent.global_step}")
pulse()
