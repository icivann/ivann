<template>
  <div
    :id="data.id"
    :class="classes"
    :style="styles"
  >
    <div
      class="__title"
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

    <div class="__content">

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

@Component({
  components: { ArrowButton },
})
export default class CustomNode extends Components.Node {
  private shouldShowOptions = false;

  toggleShouldShowOptions() {
    this.shouldShowOptions = !this.shouldShowOptions;
  }
}
</script>

<style>
  #arrow-button {
    position: absolute;
    top: 5px;
    right: 0;
    padding-right: 10px;
  }
</style>
