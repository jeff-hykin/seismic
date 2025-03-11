import { Elemental, passAlongProps } from "https://esm.sh/gh/jeff-hykin/elemental@0.6.5/main/deno.js"
import { css, components, Column, Row, askForFiles, Code, Input, Button, Checkbox, Dropdown, popUp, cx, } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/elements.js"
import { fadeIn, fadeOut } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/animations.js"
import { showToast } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/actions.js"
import { addDynamicStyleFlags, setupStyles, createCssClass, setupClassStyles, hoverStyleHelper, combineClasses, mergeStyles, AfterSilent, removeAllChildElements } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/helpers.js"
import { zip, enumerate, count, permute, combinations, wrapAroundGet } from "https://esm.sh/gh/jeff-hykin/good-js@1.13.5.1/source/array.js"

import storageObject from "https://esm.sh/gh/jeff-hykin/storage-object@0.0.3.5/main.js"
import { makeDraggable } from "./generic_tools.js"

import { Event, trigger, everyTime, once } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/events.js"
import { pointsToFunction } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/flattened/points_to_function.js"
import { fabric, FabricCanvas } from "./fabric.js"
globalThis.fabric = fabric // debugging

// 
// node
// 
    const hslBlueHue = 213
    const hslYellowHue = 49
    const hslRedHue = 14
    const energyToHue = pointsToFunction({
        xValues: [0, 1],
        yValues: [hslBlueHue, hslYellowHue],
        areSorted: true,
        method: "linearInterpolation",
    })
    const pulseAnimation = [
        [
            { transform: 'scale(1)', opacity: 1 },  // Initial state
            { transform: 'scale(1.2)', opacity: 0.8 }, // Pulsed state (enlarged and slightly faded)
            { transform: 'scale(1)', opacity: 1 },  // Back to original state
        ],
        {
            duration: 600,
            iterations: 1,
            easing: 'ease-in-out',
        }
    ]
    function Node({ label, x, y, threshold=1, startingEnergy=0, energyDecayRate=0.1, onPositionChange, ...props }) {
        const element = document.createElement("div")
        Object.assign(element.style, {
            backgroundColor: `hsl(${hslBlueHue}, 100%, 50%)`,
            color: "white",
            position: "absolute",
            left: `${x}px`,
            top: `${y}px`,
            width: "100px",
            height: "100px",
            borderRadius: "5rem",
            border: "5px solid lightgray",
            cursor: "pointer",
            transition: "background-color 0.2s ease",
        })
        Object.assign(element, {
            outputEvent: new Event(),
            _inputEvents: new Set(),
            energy: startingEnergy,
            threshold,
            energyDecayRate,
            _updateColorBasedOnEnergy: (energy)=>{
                const relativeEnergy = energy / element.threshold
                if (energy >= 1) {
                    element.style.backgroundColor = `hsl(${hslRedHue}, 100%, 50%)`
                    let animation = element.animate(
                        ...pulseAnimation
                    )
                    animation.onfinish = ()=>{
                        // reset to min color
                        element._updateColorBasedOnEnergy(Node.energyAfterFiring)
                    }
                } else {
                    element.style.backgroundColor = `hsl(${energyToHue(relativeEnergy)}, 100%, 50%)`
                }
            },
            _updateTickCallback: ()=>{
                console.debug(`element.energy is:`,element.energy)
                if (element.energy > element.threshold) {
                    trigger(element.outputEvent)
                    element._updateColorBasedOnEnergy(element.threshold+1) // show red and pulse
                    element.energy = Node.energyAfterFiring
                } else {
                    element.energy -= element.energyDecayRate
                    if (element.energy < Node.stableEnergyLevel) {
                        element.energy = Node.stableEnergyLevel
                    }
                }
            },
            _inputResponse: (input)=>{
                element.energy += input.intensity
                element._updateColorBasedOnEnergy(element.energy)
            },
            addInput: (input, intensity) => {
                everyTime(input).then(element._inputResponse)
                element._inputEvents.add(input)
            },
            removeInput: (input) => {
                // disconnect hook
                input.remove(element._inputResponse)
                // remove from set
                element._inputEvents.delete(input)
            },
        })
        makeDraggable(
            element,
            {
                onDrag: ({isStart, isEnd, x, y})=>{
                    if (onPositionChange) {
                        onPositionChange({isStart, isEnd, x, y})
                    }
                }, 
                itsPositionedCorrectlyIPromise: true,
            }
        )
        Node.allNodes.add(new WeakRef(element))
        return passAlongProps(element, props)
    }
    globalThis.Node = Node
    Object.assign(Node, {
        allNodes: new Set(),
        updateInterval: 1000,
        energyAfterFiring: 0,
        stableEnergyLevel: 0.1,
    })
    
    class FabricNode extends fabric.Circle {
        static get type() {
            return "FabricNode";
        }
        constructor({ label, left=100, top=100, radius=50, spikeThreshold=1, startingEnergy=0, energyDecayRate=0.1, id, ...custom }) {
            super({
                radius,
                "startAngle": 0,
                "endAngle": 360,
                "counterClockwise": false,
                "type": "Circle",
                "version": "6.6.1",
                "originX": "left",
                "originY": "top",
                left,
                top,
                // "width": 100,
                // "height": 100,
                "fill": `hsl(${hslBlueHue}, 100%, 50%)`,
                "stroke": "lightgray",
                "strokeWidth": 5,
                "strokeDashArray": null,
                "strokeLineCap": "butt",
                "strokeDashOffset": 0,
                "strokeLineJoin": "miter",
                "strokeUniform": false,
                "strokeMiterLimit": 4,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": false,
                "flipY": false,
                "opacity": 1,
                "shadow": null,
                "visible": true,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                ...custom,
            })
            Object.assign(this, {
                id: id||`${Math.random()}`,
                label,
                spikeThreshold,
                startingEnergy,
                energyDecayRate,
            })
        }
        toObject(propertiesToInclude) {
            return {
                ...super.toObject(propertiesToInclude),
                id: this.id,
                label: this.label,
                spikeThreshold: this.spikeThreshold,
                startingEnergy: this.startingEnergy,
                energyDecayRate: this.energyDecayRate,
            }
        }
    }
    fabric.classRegistry.setClass(FabricNode)
    
    // specific for nodes
    function NodeCanvas({
        width,
        height,
        backgroundColor,
        selectionColor,
        selectionLineWidth,
        onceFabricLoads,
        // onMouseDown,
        // onObjectAdded,
        // onAfterRender,
        // onMouseMove,
        // onMouseUp,
        // onBeforeSelection,
        // onSelectionCreated,
        // onSelectionCleared,
        // onObjectModified,
        // onObjectSelected,
        // onObjectMoving,
        // onObjectScaling,
        // onObjectRotating,
        // onObjectRemoved,
        ...props
    }={}) {
        return FabricCanvas({
            width,
            height,
            backgroundColor,
            selectionColor,
            selectionLineWidth,
            onceFabricLoads:(canvas)=>{
                globalThis.canvas = canvas // debugging only
                // TODO: add pan/zoom
                onceFabricLoads&&onceFabricLoads(canvas)
            },
            ...props,
            onMouseDown: (event)=>{
                const { target } = event
                if (target) {
                    console.debug(`target is:`,target)
                }
            },
            jsonObjects: {
                "objects": [
                    {
                        "type": "rect",
                        "left": 50,
                        "top": 50,
                        "width": 20,
                        "height": 20,
                        "fill": "green",
                        "overlayFill": null,
                        "stroke": null,
                        "strokeWidth": 1,
                        "strokeDashArray": null,
                        "scaleX": 1,
                        "scaleY": 1,
                        "angle": 0,
                        "flipX": false,
                        "flipY": false,
                        "opacity": 1,
                        "selectable": true,
                        "hasControls": true,
                        "hasBorders": true,
                        "hasRotatingPoint": false,
                        "transparentCorners": true,
                        "perPixelTargetFind": false,
                        "rx": 0,
                        "ry": 0
                    },
                    {
                        "type": "circle",
                        "left": 100,
                        "top": 100,
                        "width": 100,
                        "height": 100,
                        "fill": "red",
                        "overlayFill": null,
                        "stroke": "rgba(100,200,200)",
                        "strokeWidth": 5,
                        "strokeDashArray": null,
                        "scaleX": 1,
                        "scaleY": 1,
                        "angle": 0,
                        "flipX": false,
                        "flipY": false,
                        "opacity": 1,
                        "selectable": true,
                        "hasControls": true,
                        "hasBorders": true,
                        "hasRotatingPoint": false,
                        "transparentCorners": true,
                        "perPixelTargetFind": false,
                        "radius": 50
                    },
                    (new FabricNode({ label: "A", x:100, y:100, size:50, spikeThreshold:1, startingEnergy:0, energyDecayRate:0.1 })).toObject(),
                ],
                // "background": "rgba(0, 0, 0, 0)",
            }
        })
    }
    globalThis.canvasElement = NodeCanvas({ backgroundColor: "whitesmoke" }) // debugging
    
    // 
    // Main node updater
    // 
    setInterval(() => {
        for (let eachRef of Node.allNodes) {
            let each = eachRef.deref()
            // remove garbage collected nodes (if we didnt use WeakRefs, this would be a memory leak)
            if (!each) {
                Node.allNodes.delete(eachRef)
            }
            // have nodes update visuals in unison
            each._updateTickCallback()
        }
    }, Node.updateInterval)




const { html } = Elemental({
    ...components,
    Node,
    FabricCanvas,
})

document.body = html`
    <body font-size=15px background-color=whitesmoke overflow=none width=100vw height=100vh overflow=hidden>
        ${globalThis.canvasElement}
    </body>
`
        // <Node label="A" x=100 y=100></Node>