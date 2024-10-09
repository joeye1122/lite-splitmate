import asyncio
import argparse
from bleak import BleakScanner, BleakClient

# Constants
LOW_POW = b"\x10"
MID_POW = b"\x11"
HIGH_POW = b"\x12"
SERVICE_ID = "0000f00d-1212-efde-1523-785fef13d123"
CH_COMMAND = "00000000-1212-efde-1523-785fef13d123"
CH_STATUS = "00000001-1212-efde-1523-785fef13d123"
CH_POWER = "00000010-1212-efde-1523-785fef13d123"


def handle_disconnect(_: BleakClient):
    pass


async def get_devices(service_id):
    print("Scanning for 5 seconds, please wait...")
    devices = await BleakScanner.discover(service_uuids=[service_id])
    addresses = [device.address for device in devices]
    return addresses


async def connect_and_execute(mac_address, service_id, characteristic_id, mode):
    try:
        async with BleakClient(
            address_or_ble_device=mac_address,
            disconnected_callback=handle_disconnect,
        ) as client:
            service = client.services.get_service(service_id)
            if service:
                characteristic = service.get_characteristic(characteristic_id)
                if characteristic:
                    await client.write_gatt_char(characteristic, mode, response=False)
                else:
                    print(f"Characteristic {characteristic_id} not found.")
            else:
                print(f"Service {service_id} not found.")
    except asyncio.CancelledError:
        print(f"Device {mac_address} disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")


async def connect_and_read(mac_address, service_id, characteristic_id):
    try:
        async with BleakClient(
            address_or_ble_device=mac_address,
            disconnected_callback=handle_disconnect,
        ) as client:
            service = client.services.get_service(service_id)
            if service:
                characteristic = service.get_characteristic(characteristic_id)
                if characteristic:
                    value = await client.read_gatt_char(characteristic)
                    power_reading = 10
                    power_mode = "low"
                    if value == LOW_POW:
                        power_reading = 10
                        power_mode = "low"
                    elif value == MID_POW:
                        power_reading = 500
                        power_mode = "mid"
                    elif value == HIGH_POW:
                        power_reading = 1000
                        power_mode = "high"
                    if characteristic_id == CH_POWER:
                        return power_reading
                    elif characteristic_id == CH_STATUS:
                        return power_mode
                else:
                    print(f"Characteristic {characteristic_id} not found.")
            else:
                print(f"Service {service_id} not found.")
    except asyncio.CancelledError:
        print(f"Device {mac_address} disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")


async def main(args):
    if args.characteristic == "command":
        args.characteristic_id = CH_COMMAND
    elif args.characteristic == "status":
        args.characteristic_id = CH_STATUS
    elif args.characteristic == "power":
        args.characteristic_id = CH_POWER
    else:
        raise ValueError("--characteristic must be command, status, or power")

    if args.command == "write" and not args.power_mode:
        raise ValueError("--power_mode is required for write command")
    if args.command == "write":
        if args.power_mode == "low":
            args.power_mode = LOW_POW
        elif args.power_mode == "mid":
            args.power_mode = MID_POW
        elif args.power_mode == "high":
            args.power_mode = HIGH_POW
        else:
            raise ValueError("--power_mode must be low, mid, or high")

        await connect_and_execute(
            mac_address=args.mac_address,
            service_id=SERVICE_ID,
            characteristic_id=args.characteristic_id,
            mode=args.power_mode,
        )

    elif args.command == "read":
        result = await connect_and_read(
            mac_address=args.mac_address,
            service_id=SERVICE_ID,
            characteristic_id=args.characteristic_id,
        )
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="BLE Device Interaction")
    parser.add_argument(
        "--mac_address", required=True, help="MAC address of the BLE device"
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=["read", "write"],
        help="Command to execute (read or write)",
    )
    parser.add_argument(
        "--characteristic",
        required=True,
        choices=["command", "status", "power"],
        help="Characteristic to interact with",
    )
    parser.add_argument(
        "--power_mode",
        required=False,
        help="Power mode to write to the characteristic",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(main(args))
    print(result)
