#!/usr/bin/env python3

import argparse
import boto3
import botocore
import sys
import json
import datetime
import time

# --- Helper: retry wrapper for network timeouts/connection errors ---
def retry_once_on_network(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (botocore.exceptions.EndpointConnectionError,
                botocore.exceptions.ConnectTimeoutError,
                botocore.exceptions.ReadTimeoutError,
                botocore.exceptions.ConnectionClosedError) as e:
            print(f"[WARNING] Network error: {type(e).__name__} - retrying once...")
            try:
                time.sleep(1)
                return fn(*args, **kwargs)
            except Exception as e2:
                print(f"[ERROR] Network retry failed: {e2}")
                raise
    return wrapper

# --- Permission check helpers ---
def is_access_denied(e: botocore.exceptions.ClientError):
    code = ""
    try:
        code = e.response.get("Error", {}).get("Code", "")
    except Exception:
        pass
    return code in (
        "AccessDenied",
        "AccessDeniedException",
        "UnauthorizedOperation",
        "UnauthorizedOperationException",
        "NotAuthorized",
        "AccessDeniedError",
        "InvalidClientTokenId",
        "AuthFailure"
    )

# --- Core inspectors ---

@retry_once_on_network
def get_account_info(sts_client):
    resp = sts_client.get_caller_identity()
    account_id = resp.get("Account")
    user_arn = resp.get("Arn")
    return {"account_id": account_id, "user_arn": user_arn}

@retry_once_on_network
def collect_iam_users(iam_client):
    users = []
    paginator = iam_client.get_paginator("list_users")
    for page in paginator.paginate():
        for u in page.get("Users", []):
            username = u.get("UserName")
            user_obj = {
                "username": username,
                "user_id": u.get("UserId"),
                "arn": u.get("Arn"),
                "create_date": u.get("CreateDate").strftime("%Y-%m-%dT%H:%M:%SZ") if u.get("CreateDate") else None,
                "last_activity": None,
                "attached_policies": []
            }
            if "PasswordLastUsed" in u and u["PasswordLastUsed"]:
                user_obj["last_activity"] = u["PasswordLastUsed"].strftime("%Y-%m-%dT%H:%M:%SZ")

            try:
                uu = iam_client.get_user(UserName=username).get("User", {})
                if "PasswordLastUsed" in uu and uu["PasswordLastUsed"]:
                    user_obj["last_activity"] = uu["PasswordLastUsed"].strftime("%Y-%m-%dT%H:%M:%SZ")
            except botocore.exceptions.ClientError as e:
                if is_access_denied(e):
                    print("[WARNING] Access denied for IAM operations - skipping user enumeration")
                else:
                    raise

            try:
                ap = iam_client.list_attached_user_policies(UserName=username)
                attached = []
                for pol in ap.get("AttachedPolicies", []):
                    attached.append({
                        "policy_name": pol.get("PolicyName"),
                        "policy_arn": pol.get("PolicyArn")
                    })
                user_obj["attached_policies"] = attached
            except botocore.exceptions.ClientError as e:
                if is_access_denied(e):
                    print("[WARNING] Access denied for iam:ListAttachedUserPolicies - skipping policies")
                else:
                    raise
            users.append(user_obj)
    return users

@retry_once_on_network
def collect_ec2_instances(ec2_client):
    instances = []
    paginator = ec2_client.get_paginator("describe_instances")
    all_instances = []
    for page in paginator.paginate():
        for r in page.get("Reservations", []):
            for inst in r.get("Instances", []):
                all_instances.append(inst)

    ami_ids = list({i.get("ImageId") for i in all_instances if i.get("ImageId")})
    ami_map = {}
    if ami_ids:
        try:
            images = ec2_client.describe_images(ImageIds=ami_ids).get("Images", [])
            for img in images:
                ami_map[img.get("ImageId")] = img.get("Name")
        except botocore.exceptions.ClientError as e:
            if is_access_denied(e):
                print("[WARNING] Access denied for ec2:DescribeImages - AMI names will be empty")
            else:
                raise

    for inst in all_instances:
        instance_id = inst.get("InstanceId")
        state = inst.get("State", {}).get("Name")
        public_ip = inst.get("PublicIpAddress")
        private_ip = inst.get("PrivateIpAddress")
        az = inst.get("Placement", {}).get("AvailabilityZone")
        launch_time = inst.get("LaunchTime")
        launch_time_str = launch_time.strftime("%Y-%m-%dT%H:%M:%SZ") if launch_time else None
        ami_id = inst.get("ImageId")
        ami_name = ami_map.get(ami_id)
        security_groups = [sg.get("GroupId") for sg in inst.get("SecurityGroups", []) if sg.get("GroupId")]
        tags = {t.get("Key"): t.get("Value") for t in inst.get("Tags", [])}
        instance_obj = {
            "instance_id": instance_id,
            "instance_type": inst.get("InstanceType"),
            "state": state,
            "public_ip": public_ip if public_ip else None,
            "private_ip": private_ip if private_ip else None,
            "availability_zone": az,
            "launch_time": launch_time_str,
            "ami_id": ami_id,
            "ami_name": ami_name,
            "security_groups": security_groups,
            "tags": tags
        }
        instances.append(instance_obj)
    return instances

@retry_once_on_network
def collect_s3_buckets(s3_client, sts_account_id, region_hint=None):
    buckets = []
    try:
        resp = s3_client.list_buckets()
    except botocore.exceptions.ClientError as e:
        if is_access_denied(e):
            print("[WARNING] Access denied for S3 operations - skipping bucket enumeration")
            return []
        else:
            raise
    for b in resp.get("Buckets", []):
        bucket_name = b.get("Name")
        create_date = b.get("CreationDate")
        create_str = create_date.strftime("%Y-%m-%dT%H:%M:%SZ") if create_date else None

        try:
            loc = s3_client.get_bucket_location(Bucket=bucket_name)
            region = loc.get("LocationConstraint") or "us-east-1"
            if region == "EU":
                region = "eu-west-1"
        except botocore.exceptions.ClientError as e:
            if is_access_denied(e):
                print(f"[ERROR] Failed to access S3 bucket '{bucket_name}': Access Denied")
                region = region_hint or "unknown"
            else:
                raise

        regional_s3 = boto3.client("s3", region_name=region) if region else s3_client

        object_count = 0
        size_bytes = 0
        try:
            paginator = regional_s3.get_paginator("list_objects_v2")
            page_iter = paginator.paginate(Bucket=bucket_name)
            for page in page_iter:
                contents = page.get("Contents", [])
                object_count += len(contents)
                for obj in contents:
                    size_bytes += obj.get("Size", 0)
        except botocore.exceptions.ClientError as e:
            if is_access_denied(e):
                print(f"[ERROR] Failed to access S3 bucket '{bucket_name}': Access Denied")
                object_count = 0
                size_bytes = 0
            else:
                raise

        buckets.append({
            "bucket_name": bucket_name,
            "creation_date": create_str,
            "region": region,
            "object_count": object_count,
            "size_bytes": size_bytes
        })
    return buckets

@retry_once_on_network
def collect_security_groups(ec2_client):
    groups = []
    resp = ec2_client.describe_security_groups()
    for sg in resp.get("SecurityGroups", []):
        group_id = sg.get("GroupId")
        group_name = sg.get("GroupName")
        desc = sg.get("Description")
        vpc_id = sg.get("VpcId")
        inbound = []
        for p in sg.get("IpPermissions", []):
            proto = p.get("IpProtocol")
            proto_label = "all" if proto == "-1" else proto
            fromp = p.get("FromPort")
            top = p.get("ToPort")
            prange = "all" if fromp is None or top is None else f"{fromp}-{top}" if fromp != top else f"{fromp}-{top}"
            sources = [r.get("CidrIp") or r.get("CidrIpv6") or r.get("GroupId") for r in p.get("IpRanges", []) + p.get("Ipv6Ranges", []) + p.get("UserIdGroupPairs", [])]
            if not sources:
                sources = ["-"]
            for s in sources:
                inbound.append({"protocol": proto_label, "port_range": prange, "source": s})

        outbound = []
        for p in sg.get("IpPermissionsEgress", []):
            proto = p.get("IpProtocol")
            proto_label = "all" if proto == "-1" else proto
            fromp = p.get("FromPort")
            top = p.get("ToPort")
            prange = "all" if fromp is None or top is None else f"{fromp}-{top}" if fromp != top else f"{fromp}-{top}"
            destinations = [r.get("CidrIp") or r.get("CidrIpv6") or r.get("GroupId") for r in p.get("IpRanges", []) + p.get("Ipv6Ranges", []) + p.get("UserIdGroupPairs", [])]
            if not destinations:
                destinations = ["-"]
            for d in destinations:
                outbound.append({"protocol": proto_label, "port_range": prange, "destination": d})

        groups.append({
            "group_id": group_id,
            "group_name": group_name,
            "description": desc,
            "vpc_id": vpc_id,
            "inbound_rules": inbound,
            "outbound_rules": outbound
        })
    return groups

# --- Output formatting ---

def format_json(output_obj, outfile=None):
    text = json.dumps(output_obj, indent=2, default=str)
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)

def safe_truncate(s, width):
    s = "" if s is None else str(s)
    if len(s) <= width:
        return s
    return s[: width - 3] + "..."

def format_table(output_obj, outfile=None):
    acc = output_obj.get("account_info", {})
    resources = output_obj.get("resources", {})
    summary = output_obj.get("summary", {})

    lines = []
    lines.append(f"AWS Account: {acc.get('account_id')} ({acc.get('region')})")
    lines.append(f"Scan Time: {acc.get('scan_timestamp')}")
    lines.append("")

    # IAM USERS
    iam_users = resources.get("iam_users", [])
    lines.append(f"IAM USERS ({len(iam_users)} total)")
    if iam_users:
        lines.append(f"{'Username':20} {'Create Date':20} {'Last Activity':20} {'Policies'}")
        for u in iam_users:
            uname = safe_truncate(u.get("username", ""), 20)
            cdate = safe_truncate(u.get("create_date", "")[:10], 20) if u.get("create_date") else "-"
            lact = safe_truncate(u.get("last_activity", "")[:10], 20) if u.get("last_activity") else "-"
            polcount = len(u.get("attached_policies", []))
            lines.append(f"{uname:20} {cdate:20} {lact:20} {polcount}")
    else:
        lines.append("[No IAM users found]")
    lines.append("")

    # EC2 INSTANCES
    ec2_instances = resources.get("ec2_instances", [])
    running = sum(1 for i in ec2_instances if i.get("state") == "running")
    total = len(ec2_instances)
    lines.append(f"EC2 INSTANCES ({running} running, {total - running} stopped/other)")
    if ec2_instances:
        lines.append(f"{'Instance ID':20} {'Type':10} {'State':10} {'Public IP':16} {'Launch Time'}")
        for i in ec2_instances:
            iid = safe_truncate(i.get("instance_id", ""), 20)
            itype = safe_truncate(i.get("instance_type", ""), 10)
            state = safe_truncate(i.get("state", ""), 10)
            pub = safe_truncate(i.get("public_ip") or "-", 16)
            lt = safe_truncate(i.get("launch_time") or "-", 19)
            lines.append(f"{iid:20} {itype:10} {state:10} {pub:16} {lt}")
    else:
        print(f"[WARNING] No EC2 instances found in '{acc.get('region')}'")
    lines.append("")

    # S3 BUCKETS
    s3b = resources.get("s3_buckets", [])
    lines.append(f"S3 BUCKETS ({len(s3b)} total)")
    if s3b:
        lines.append(f"{'Bucket Name':25} {'Region':12} {'Created':12} {'Objects':8} {'Size (MB)'}")
        for b in s3b:
            name = safe_truncate(b.get("bucket_name", ""), 25)
            region = safe_truncate(b.get("region", ""), 12)
            created = safe_truncate((b.get("creation_date") or "")[:10], 12)
            objs = b.get("object_count") or "-"
            size_mb = b.get("size_bytes", 0)
            try:
                size_mb_text = f"~{(size_mb/1024/1024):.1f}"
            except Exception:
                size_mb_text = "-"
            lines.append(f"{name:25} {region:12} {created:12} {objs:8} {size_mb_text}")
    else:
        lines.append(f"[WARNING] No S3 buckets found in '{acc.get('region')}'")
    lines.append("")

    # SECURITY GROUPS
    sgs = resources.get("security_groups", [])
    lines.append(f"SECURITY GROUPS ({len(sgs)} total)")
    if sgs:
        lines.append(f"{'Group ID':15} {'Name':20} {'VPC ID':15} {'Inbound Rules'}")
        for g in sgs:
            gid = safe_truncate(g.get("group_id", ""), 15)
            name = safe_truncate(g.get("group_name", ""), 20)
            vpc = safe_truncate(g.get("vpc_id") or "-", 15)
            inbound_count = len(g.get("inbound_rules", []))
            lines.append(f"{gid:15} {name:20} {vpc:15} {inbound_count}")
    else:
        lines.append("[No Security Groups found]")

    out_text = "\n".join(lines)
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        print(out_text)

# --- Main runner ---
def main():
    parser = argparse.ArgumentParser(description="AWS Inspector - collect IAM, EC2, S3, Security Groups")
    parser.add_argument("--region", help="AWS region to inspect (default: from credentials/config)", default=None)
    parser.add_argument("--output", help="Output file path (default: stdout)", default=None)
    parser.add_argument("--format", help="Output format: json or table (default: json)", default="json", choices=["json", "table"])
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region) if args.region else boto3.Session()
    region_to_use = args.region or session.region_name

    if region_to_use:
        available = boto3.session.Session().get_available_regions("ec2")
        if region_to_use not in available:
            print(f"[ERROR] Invalid or unknown region '{region_to_use}' - aborting")
            sys.exit(1)

    try:
        sts_client = session.client("sts")
        acc_info = get_account_info(sts_client)
    except botocore.exceptions.NoCredentialsError:
        print("[ERROR] No AWS credentials found. Configure via 'aws configure' or environment variables.")
        sys.exit(2)
    except botocore.exceptions.ClientError as e:
        print(f"[ERROR] Failed to validate AWS credentials: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Unexpected error validating credentials: {e}")
        sys.exit(2)

    scan_ts = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat() + "Z"

    iam_client = session.client("iam")
    ec2_client = session.client("ec2", region_name=region_to_use) if region_to_use else session.client("ec2")
    s3_client = session.client("s3", region_name=region_to_use) if region_to_use else session.client("s3")

    resources = {
        "iam_users": [],
        "ec2_instances": [],
        "s3_buckets": [],
        "security_groups": []
    }

    try:
        resources["iam_users"] = collect_iam_users(iam_client)
    except botocore.exceptions.ClientError as e:
        if is_access_denied(e):
            print("[WARNING] Access denied for IAM operations - skipping user enumeration")

    try:
        resources["ec2_instances"] = collect_ec2_instances(ec2_client)
    except botocore.exceptions.ClientError as e:
        if is_access_denied(e):
            print("[WARNING] Access denied for EC2 operations - skipping instance enumeration")

    try:
        resources["s3_buckets"] = collect_s3_buckets(s3_client, acc_info.get("account_id"), region_hint=region_to_use)
    except botocore.exceptions.ClientError as e:
        if is_access_denied(e):
            print("[WARNING] Access denied for S3 operations - skipping bucket enumeration")

    try:
        resources["security_groups"] = collect_security_groups(ec2_client)
    except botocore.exceptions.ClientError as e:
        if is_access_denied(e):
            print("[WARNING] Access denied for EC2 security-group operations - skipping security group enumeration")
        else:
            print("[WARNING] Failed Security Group enumeration")

    summary = {
        "total_users": len(resources["iam_users"]),
        "running_instances": sum(1 for i in resources["ec2_instances"] if i.get("state") == "running"),
        "total_buckets": len(resources["s3_buckets"]),
        "security_groups": len(resources["security_groups"])
    }

    output_obj = {
        "account_info": {
            "account_id": acc_info.get("account_id"),
            "user_arn": acc_info.get("user_arn"),
            "region": region_to_use or "unknown",
            "scan_timestamp": scan_ts
        },
        "resources": resources,
        "summary": summary
    }

    fmt = args.format.lower()
    if fmt == "json":
        format_json(output_obj, outfile=args.output)
    else:
        format_table(output_obj, outfile=args.output)

if __name__ == "__main__":
    main()
