import asyncio
import websockets
import json
import torch


async def talker_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Identify itself as a talker
        await websocket.send("talker")

        while True:
            tensor_data = 2 * torch.rand((2, 8), device='cuda:0') - 1
            # Convert tensor to list and then to JSON
            data_to_send = json.dumps(tensor_data.cpu().numpy().tolist())
            await websocket.send(data_to_send)

            print("Sent:", data_to_send)
            await asyncio.sleep(1)  # Send every second

# Run the talker client
asyncio.get_event_loop().run_until_complete(talker_client())
