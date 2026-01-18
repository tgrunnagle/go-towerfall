# Architecture Documentation

This directory contains detailed architecture documentation for the go-towerfall project, a Towerfall-inspired arena combat game with ML bot training capabilities.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                    │
│                    React UI + Canvas Game Engine                         │
│                         (WebSocket client)                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    HTTP REST API │ WebSocket /ws
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                              Backend                                     │
│                   Go Game Server (authoritative)                         │
│              Physics simulation, game state, matchmaking                 │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                         HTTP     │     HTTP
                      spawn bots  │     training & play
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                               Bot                                        │
│                 Python ML Training Framework (PPO)                       │
│            Reinforcement learning, self-play, model registry             │
└─────────────────────────────────────────────────────────────────────────┘

Backend ← Bot: Training runs, playing in human games (Bot initiates HTTP requests)
Backend → Bot: Spawning bot processes as opponents
```

## Component Documentation

### [Backend Architecture](backend.md)

Go-based authoritative game server handling real-time physics simulation, WebSocket communication, and HTTP APIs.

**Key areas covered:**
- Server architecture and concurrency model
- Game room tick loop and event-driven design
- GameObject interface (Player, Arrow, Block)
- Physics constants and collision detection
- HTTP/WebSocket communication protocols
- Training mode configuration and spectator throttling

### [Bot Architecture](bot.md)

Python-based reinforcement learning framework for training TowerFall bots using PPO (Proximal Policy Optimization).

**Key areas covered:**
- Neural network architecture (ActorCriticNetwork)
- PPO training with GAE advantages
- Gymnasium environment integration
- Observation space (414 features) and action space (27 discrete actions)
- Successive self-play training pipeline
- Model registry and metrics logging

### [Frontend Architecture](frontend.md)

Hybrid web client combining React for UI/routing with a vanilla JavaScript game engine for real-time canvas rendering.

**Key areas covered:**
- React + Canvas hybrid design pattern
- WebSocket message handling and reconnection
- Game state management and object lifecycle
- Position interpolation and client-side prediction
- Rendering pipeline and animation system
- Build configuration and environment setup

## Quick Reference

| Component | Language | Primary Role |
|-----------|----------|--------------|
| Backend | Go | Authoritative game server, physics, matchmaking |
| Bot | Python | ML training, PPO agent, model management |
| Frontend | JS/React | Browser client, rendering, user input |

## Getting Started

For development setup and running instructions, see the root [README.md](../../README.md).
