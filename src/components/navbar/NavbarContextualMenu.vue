<template>
    <div id="contextual-menu">
      <div v-for="(editor, index) in editors" :key="index">
        <VerticalMenuButton
          :label="editor.name"
          :onClick="() => switchEditor({ editorType, index})"
          :isSelected="editorType === currEditorType && index === currEditorIndex"
        />
      </div>
      <VerticalMenuButton
        :label="'+'"
        :onClick="this.createNewEditor"
        :isSelected="false"
      />
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import VerticalMenuButton from '@/components/buttons/VerticalMenuButton.vue';
import { mapGetters, mapMutations } from 'vuex';
import { Editor } from '@baklavajs/core';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';

@Component({
  components: { VerticalMenuButton },
  computed: mapGetters([
    'currEditorType',
    'currEditorIndex',
  ]),
  methods: mapMutations(['switchEditor']),
})
export default class NavbarContextualMenu extends Vue {
  @Prop({ required: true }) readonly editors!: Editor[];
  @Prop({ required: true }) readonly editorType!: EditorType
  @Getter('editorNames') editorNames!: Set<string>;
  @Mutation('newEditor') newEditor!: (arg0: { editorType: EditorType; name: string}) => void;

  private createNewEditor(): void {
    let isNameUnique = false;
    while (!isNameUnique) {
      const name = prompt('Please enter a unique name for the editor');

      // Name is null if cancelled
      if (name === null) break;

      // Loop until unique non-empty name has been entered
      if (name !== '' && !this.editorNames.has(name)) {
        isNameUnique = true;
        this.newEditor({ editorType: this.editorType, name });
      }
    }
  }
}

</script>

<style scoped>
  #contextual-menu {
    border: 1px solid var(--grey);
  }
</style>
