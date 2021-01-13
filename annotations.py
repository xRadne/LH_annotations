import json
import torch


room_index = {"Alcove": 11,
              "Attic": 11,
              "Ballroom": 11,
              "Bar": 11,
              "Basement": 11,
              "Bath": 6,
              "Bedroom": 5,
              "Bed Room": 5,
              "CarPort": 10,
              "Church": 11,
              "Closet": 9,
              "ConferenceRoom": 11,
              "Conservatory": 11,
              "Counter": 11,
              "Den": 11,
              "Dining": 4,
              "DraughtLobby": 7,
              "DressingRoom": 9,
              "EatingArea": 4,
              "Elevated": 11,
              "Elevator": 11,
              "Entry": 7,
              "ExerciseRoom": 11,
              "Garage": 10,
              "Garbage": 11,
              "Hall": 11,
              "HallWay": 7,
              "HotTub": 11,
              "Kitchen": 3,
              "Library": 11,
              "LivingRoom": 4,
              "Living Room": 4,
              "Loft": 11,
              "Lounge": 4,
              "MediaRoom": 11,
              "MeetingRoom": 11,
              "Museum": 11,
              "Nook": 11,
              "Office": 11,
              "OpenToBelow": 11,
              "Outdoor": 1,
              "Pantry": 11,
              "Reception": 11,
              "RecreationRoom": 11,
              "RetailSpace": 11,
              "Room": 11,
              "Sanctuary": 11,
              "Sauna": 6,
              "ServiceRoom": 11,
              "ServingArea": 11,
              "Skylights": 11,
              "Stable": 11,
              "Stage": 11,
              "StairWell": 11,
              "Storage": 9,
              "SunRoom": 11,
              "SwimmingPool": 11,
              "TechnicalRoom": 11,
              "Theatre": 11,
              "Undefined": 11,
              "UserDefined": 11,
              "Utility": 11,
              "Background": 0,  # Not in data. The default outside label
              "Wall": 2,
              "WallSegmentation": 2,
              "Railing": 8}

icon_index = {"Window": 1,
              "Door": 2,
              "Closet": 3,
              "ClosetRound": 3,
              "ClosetTriangle": 3,
              "CoatCloset": 3,
              "CoatRack": 3,
              "CounterTop": 3,
              "Housing": 3,
              "ElectricalAppliance": 4,
              "Electrical Applience": 4,
              "WoodStove": 4,
              "GasStove": 4,
              "Toilet": 5,
              "Urinal": 5,
              "SideSink": 6,
              "Sink": 6,
              "RoundSink": 6,
              "CornerSink": 6,
              "DoubleSink": 6,
              "DoubleSinkRight": 6,
              "WaterTap": 6,
              "SaunaBenchHigh": 7,
              "SaunaBenchLow": 7,
              "SaunaBenchMid": 7,
              "SaunaBench": 7,
              "Sauna Bench": 7,
              "Fireplace": 8,
              "Fire Place": 8,
              "FireplaceCorner": 8,
              "FireplaceRound": 8,
              "PlaceForFireplace": 8,
              "PlaceForFireplaceCorner": 8,
              "PlaceForFireplaceRound": 8,
              "Bathtub": 9,
              "BathtubRound": 9,
              "Chimney": 10,
              "Misc": None,
              "BaseCabinetRound": None,
              "BaseCabinetTriangle": None,
              "BaseCabinet": None,
              "WallCabinet": None,
              "Shower": None,
              "ShowerCab": None,
              "ShowerPlatform": None,
              "ShowerScreen": None,
              "ShowerScreenRoundRight": None,
              "ShowerScreenRoundLeft": None,
              "Jacuzzi": None}

sort_order = {  # Order or precedence. Highest number to be drawn on top, lower numbers will be replaced by higher overlapping annotations. 
    "Background": 0,
    "Undefined": 1,
    "Outdoor": 2,
    "Garage": 3,
    "Bath": 4,
    "Bed Room": 5,
    "Entry": 6,
    "Kitchen": 7,
    "Living Room": 8,
    "Storage": 9,
    "Railing": 10,
    "Wall": 11,
    "WallSegmentation": 12,
    "Icon": 13,
}


def custom_sort_order(obj):
    if get_class_type(obj["title"]) == "Room":
        return sort_order[get_name(obj["title"])]
    else:
        return sort_order["Icon"]


def get_class_type(class_name):
    return class_name.split(" - ")[0]


def get_name(class_name):
    return class_name.split(" - ")[1]


def get_class_index(class_name):
    class_type = get_class_type(class_name)
    name = get_name(class_name)
    if class_type == "Room":
        index = room_index[name]
    elif class_type == "Icon":
        index = icon_index[name]
    else:
        raise ValueError(f"Class type {class_type} is not valid")
    return index


def parse_labelbox_to_tensor(path, height, width):
    labelbox_label = json.load(open(path, "r"))
    labels = labelbox_label["objects"]

    labels = sorted(labels, key=custom_sort_order)

    label = torch.zeros(2, height, width)
    for obj in labels:
        class_name = obj["title"]

        if class_name == "Room - WallSegmentation":
            print(f"'Room - WallSegmentation' encountered. This is currently ignored. At {path}")
            continue
        class_index = get_class_index(class_name)
        class_type = get_class_type(class_name)
        x1, y1 = obj["bbox"]["left"], obj["bbox"]["top"]
        x2, y2 = x1 + obj["bbox"]["width"], y1 + obj["bbox"]["height"]

        if class_type == "Room":
            label[0, y1:y2, x1:x2] = class_index
        elif class_type == "Icon":
            label[1, y1:y2, x1:x2] = class_index    
    return label


if __name__ == "__main__":
    label = parse_labelbox_to_tensor(
        "./0a/fe/da49de038b01e9272e6dc962da7e/label.json", 256, 407)

    np_label = label.data.numpy()

    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]

    fig, ax = plt.subplots()
    rseg = ax.imshow(np_label[0], vmin=0, vmax=12-0.1)
    cbar = plt.colorbar(rseg, ticks=np.arange(len(room_classes)) + 0.5, fraction=0.046, pad=0.01)
    cbar.ax.set_yticklabels([f"{i}: {x}" for i, x in enumerate(room_classes)], fontsize=10)
    icons = np_label[1]
    icons[icons == 0] = np.nan
    iseg = ax.imshow(icons, vmin=0, vmax=12-0.1)
    
    plt.show()

    print("Done!")
