import pinecone
print("Pinecone version =", pinecone.__version__)

# Check if init method exists
print("init function exists =", hasattr(pinecone, "init"))

# Try to use the init method
if hasattr(pinecone, "init"):
    try:
        print("Initializing with init()...")
        pinecone.init(api_key="dummy_key")
        print("Init successful!")
    except Exception as e:
        print(f"Init failed: {str(e)}")

# Check if Pinecone class exists
print("Pinecone class exists =", hasattr(pinecone, "Pinecone"))

# Try to use the Pinecone class
if hasattr(pinecone, "Pinecone"):
    try:
        print("Initializing with Pinecone class...")
        pc = pinecone.Pinecone(api_key="dummy_key")
        print("Pinecone class initialization successful!")
    except Exception as e:
        print(f"Pinecone class initialization failed: {str(e)}")

# Try importing from submodule
try:
    print("Trying to import from submodule...")
    from pinecone.control.pinecone import Pinecone
    print("Import from submodule successful!")
    pc2 = Pinecone(api_key="dummy_key")
    print("Submodule Pinecone initialization successful!")
except Exception as e:
    print(f"Submodule import/init failed: {str(e)}") 