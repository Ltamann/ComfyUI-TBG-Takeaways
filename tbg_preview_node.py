

class TBGPreviewSenderWS:
    """
    Sends preview images via ComfyUI's built-in WebSocket
    No external server needed!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "node_name": ("STRING", {"default": "refiner_preview"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "send_preview"
    OUTPUT_NODE = True
    CATEGORY = "TBG"
    
    def send_preview(self, images, node_name="refiner_preview"):
        """
        Send preview via ComfyUI's WebSocket to all connected clients
        """
        server = PromptServer.instance
        
        # Process first image
        if len(images) > 0:
            image = images[0]
            
            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Send via ComfyUI's WebSocket with custom event type
            server.send_sync("tbg_preview", {
                "node_name": node_name,
                "image": f"data:image/png;base64,{img_base64}",
                "timestamp": str(np.datetime64('now'))
            })
            
            print(f"[TBG] Sent preview via ComfyUI WebSocket: {node_name}")
        
        # Pass through images unchanged
        return (images,)




# Node registration
NODE_CLASS_MAPPINGS = {
    "TBG_Preview_Sender_WebSocked": TBGPreviewSenderWS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TBG_Preview_Sender_WebSocked": "TBG Preview Sender (WebSocket)",
}
