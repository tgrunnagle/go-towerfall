# Example basic bot for Towerfall using GameClient to connect and interact with the game server.
import asyncio
import argparse
import logging
import sys
import websockets
from game_client import GameClient


class ExampleBot:
    def __init__(self, client: GameClient):
        self.client = client
        self.running = False
        self._logger = logging.getLogger(__name__)
        self.client.register_message_handler(self._handle_message)

    async def run(self):
        """Main bot loop"""
        self.running = True
        try:
            while self.running:
                # Your ML model inference would go here
                # For now, we'll just implement a simple example that moves back and forth

                # Move right
                await self.client.send_keyboard_input("d", True)
                await asyncio.sleep(0.5)
                await self.client.send_keyboard_input("d", False)

                # Move left
                await self.client.send_keyboard_input("a", True)
                await asyncio.sleep(0.5)
                await self.client.send_keyboard_input("a", False)

                await asyncio.sleep(0.1)  # Don't spam the server
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except Exception as e:
            self._logger.exception("Bot error", e)
            self.running = False

    def stop(self):
        """Stop the bot"""
        self.running = False

    async def _handle_message(self, message: dict) -> None:
        """Handle incoming messages"""
        # TODO: Handle messages
        pass

    async def wait_for_exit(self):
        sys.stdout.write("Press Enter to exit...")
        sys.stdout.flush()
        await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        await self.client.exit_game()


async def main(args):
    # Create a game client
    client = GameClient()

    try:
        # Connect to a game room and get the listener task
        listener_task = await client.connect(
            room_code=args.room_code,
            player_name=args.player_name,
            room_password=args.password,
        )

        # Create and run the bot
        bot = ExampleBot(client)
        bot_task = asyncio.create_task(bot.run())

        exit_task = asyncio.create_task(bot.wait_for_exit())

        # Wait for either task to complete
        await asyncio.wait(
            [listener_task, bot_task, exit_task], return_when=asyncio.FIRST_COMPLETED
        )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="ML bot for Towerfall")
    parser.add_argument(
        "--player_name", type=str, default="ML-Bot", help="Name of the bot player"
    )
    parser.add_argument(
        "--room_code", type=str, required=True, help="Room code to join"
    )
    parser.add_argument(
        "--password", type=str, default=None, help="Optional room password"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
