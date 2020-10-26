<template>
  <div
    :id="data.id"
    :class="classes"
    :style="styles"
  >
    <div
      id="header"
      :style="titleBackground"
      @mousedown.self.stop="startDrag"
      @contextmenu.self.prevent="openContextMenu"
    >

      <span v-if="!renaming">
        {{ data.name }}
         <ArrowButton
           id="arrow-button"
           :initialUp="true"
           v-on:arrow-button-clicked="toggleShouldShowOptions"
         />
      </span>
      <input
        v-else
        type="text"
        class="dark-input"
        v-model="tempName"
        placeholder="Node Name"
        v-click-outside="doneRenaming"
        @keydown.enter="doneRenaming"
      >

      <component
        :is="plugin.components.contextMenu"
        v-model="contextMenu.show"
        :x="contextMenu.x" :y="contextMenu.y"
        :items="contextMenu.items"
        @click="onContextMenu"
      ></component>
    </div>

    <div
      class="__content"
      id="content"
    >

      <!-- Outputs -->
      <div class="__outputs">
        <component
          :is="plugin.components.nodeInterface"
          v-for="(output, name) in data.outputInterfaces"
          :key="output.id"
          :name="name"
          :data="output"
        ></component>
      </div>

      <!-- Options -->
      <div v-if=shouldShowOptions class="__options">
        <template v-for="[name, option] in data.options">

          <component
            :is="plugin.components.nodeOption"
            :key="name"
            :name="name"
            :option="option"
            :componentName="option.optionComponent"
            :node="data"
            @openSidebar="openSidebar(name)"
          ></component>

          <portal :key="'sb_' + name" to="sidebar"
                  v-if="plugin.sidebar.nodeId === data.id
                        && plugin.sidebar.optionName === name
                        && option.sidebarComponent"
          >
            <component
              :is="plugin.components.nodeOption"
              :key="data.id + name"
              :name="name"
              :option="option"
              :componentName="option.sidebarComponent"
              :node="data"
            ></component>
          </portal>

        </template>
      </div>

      <!-- Inputs -->
      <div class="__inputs">
        <component
          :is="plugin.components.nodeInterface"
          v-for="(input, name) in data.inputInterfaces"
          :key="input.id"
          :name="name"
          :data="input"
        ></component>
      </div>

    </div>

  </div>
</template>

<script lang="ts">
import { Component } from 'vue-property-decorator';
import { Components } from '@baklavajs/plugin-renderer-vue';
import ArrowButton from '@/inputs/ArrowButton.vue';
import { Layers } from '@/nodes/model/Types';

@Component({
  components: { ArrowButton },
})
export default class CustomNode extends Components.Node {
  private shouldShowOptions = false;

  private toggleShouldShowOptions(): void {
    this.shouldShowOptions = !this.shouldShowOptions;
  }

  get titleBackground() {
    switch (this.data.type) {
      case Layers.Linear:
        return { background: 'var(--black)' };
      case Layers.Conv:
        return { background: 'var(--blue)' };
      case Layers.Pool:
        return { background: 'var(--red)' };
      case Layers.Regularization:
        return { background: 'var(--pink)' };
      case Layers.Reshape:
        return { background: 'var(--green)' };
      default:
        return { background: 'var(--black)' };
    }
  }
}
</script>

<style scoped lang="scss">
  #arrow-button {
    position: absolute;
    top: 5px;
    right: 0;
    padding-right: 10px;
  }

  #content {
    background: var(--dark-grey);
  }

  .node {
    /*font-family: Roboto, serif;*/
    font-size: 14px;
    &:hover {
      box-shadow: 0 0 0 0.35px var(--blue);
    }
    &.--selected {
      z-index: 5;
      box-shadow: 0 0 0 1px var(--blue);
    }
    & > #header {
      color: var(--foreground);
      padding: 0.2em 0.75em;
      border-radius: 4px 4px 0 0;
      font-size: 17px;
      text-align: center;

      & > span {
        pointer-events: none;
      }
    }
  }
</style>
