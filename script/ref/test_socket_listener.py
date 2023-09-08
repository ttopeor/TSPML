import asyncio
import websockets
import websockets.exceptions


async def listener_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Identify itself as a listener
        await websocket.send("listener")

        try:
            while True:
                # Continuously receive data from the server
                data = await websocket.recv()
                print(f"Received data from server: {data}")

        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by the server.")

        except Exception as e:
            print(f"Unexpected error: {e}")

# Run the listener client
asyncio.get_event_loop().run_until_complete(listener_client())
