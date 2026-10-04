"""Issue RTX SYLPH Full Grok keys. Run locally; do not ship generated keys in the public zip."""
import argparse
import sylph_license

parser = argparse.ArgumentParser()
parser.add_argument("--plan", default="premium", choices=("premium", "sub"))
parser.add_argument("--email", default="owner")
parser.add_argument("--activate", action="store_true", help="Write license.json in this folder")
args = parser.parse_args()
key = sylph_license.issue_key(args.plan, args.email)
print(key)
if args.activate:
    print(sylph_license.save_license(key, args.email))
