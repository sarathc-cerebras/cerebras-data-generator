import secrets

def generate_api_token(length: int = 32) -> str:
    """Generate a secure random API token."""
    return secrets.token_urlsafe(length)

if __name__ == "__main__":
    # Generate 3 tokens
    tokens = [generate_api_token() for _ in range(3)]
    
    print("Generated API Tokens:")
    print("=" * 60)
    for i, token in enumerate(tokens, 1):
        print(f"Token {i}: {token}")
    
    print("\n" + "=" * 60)
    print("\nAdd to your environment:")
    print(f'export API_AUTH_TOKENS="{",".join(tokens)}"')